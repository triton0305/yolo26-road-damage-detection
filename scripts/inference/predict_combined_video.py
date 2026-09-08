"""Create YouTube-ready videos with combined segmentation and detection overlays."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
from ultralytics import YOLO


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
DETECT_NAME_OVERRIDES = {
    "절삭보수부파손": "절삭보수부",
    "긴급보수부파손": "긴급보수부",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detect-model", type=Path, required=True)
    parser.add_argument("--segment-model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detect-conf", type=float, default=0.25)
    parser.add_argument("--segment-conf", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--bitrate", default="8M")
    return parser.parse_args()


def encoder_command(
    ffmpeg: str,
    output: Path,
    width: int,
    height: int,
    fps: float,
    bitrate: str,
) -> list[str]:
    return [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:g}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-b:v",
        bitrate,
        "-maxrate",
        "10M",
        "-bufsize",
        "16M",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def attach_audio(ffmpeg: str, video_only: Path, source: Path, output: Path) -> None:
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_only),
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "256k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip())


def process_video(
    source: Path,
    output: Path,
    detect_model: YOLO,
    segment_model: YOLO,
    args: argparse.Namespace,
    ffmpeg: str,
) -> None:
    capture = cv2.VideoCapture(str(source))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or source_fps <= 0:
        raise RuntimeError(f"Cannot open video: {source}")

    expected_frames = round(source_frames * args.fps / source_fps)
    video_only = output.with_name(f".{output.stem}.video_only.mp4")
    encoder = subprocess.Popen(
        encoder_command(
            ffmpeg,
            video_only,
            args.width,
            args.height,
            args.fps,
            args.bitrate,
        ),
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    input_index = output_index = 0
    started = time.monotonic()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_time = input_index / source_fps
            wanted_time = output_index / args.fps
            input_index += 1
            if frame_time + (0.5 / source_fps) < wanted_time:
                continue

            frame = cv2.resize(
                frame,
                (args.width, args.height),
                interpolation=cv2.INTER_AREA,
            )
            segment_result = segment_model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.segment_conf,
                classes=None,
                device=args.device,
                retina_masks=True,
                verbose=False,
            )[0]
            detect_result = detect_model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.detect_conf,
                classes=None,
                device=args.device,
                verbose=False,
            )[0]
            detect_result.names = {
                class_id: DETECT_NAME_OVERRIDES.get(name, name)
                for class_id, name in detect_result.names.items()
            }

            combined = segment_result.plot(
                boxes=True,
                masks=True,
                labels=True,
                conf=True,
            )
            combined = detect_result.plot(
                img=combined,
                boxes=True,
                masks=False,
                labels=True,
                conf=True,
            )
            if encoder.stdin is None:
                raise RuntimeError("FFmpeg stdin is unavailable")
            encoder.stdin.write(combined.tobytes())
            output_index += 1

            if output_index % 100 == 0:
                elapsed = time.monotonic() - started
                rate = output_index / elapsed
                remaining = max(expected_frames - output_index, 0) / max(rate, 0.01)
                print(
                    f"video={source.name} frames={output_index}/{expected_frames} "
                    f"fps={rate:.2f} eta_min={remaining / 60:.1f}",
                    flush=True,
                )
    finally:
        capture.release()
        if encoder.stdin is not None:
            encoder.stdin.close()

    error = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
    return_code = encoder.wait()
    if return_code:
        raise RuntimeError(error.strip())

    attach_audio(ffmpeg, video_only, source, output)
    video_only.unlink()
    print(f"completed={output} frames={output_index}", flush=True)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    videos = sorted(
        path for path in args.source.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise RuntimeError(f"No videos found: {args.source}")

    outputs = [args.output / f"{video.stem}_final_1080p30.mp4" for video in videos]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Output already exists: {existing[0]}")

    detect_model = YOLO(str(args.detect_model))
    segment_model = YOLO(str(args.segment_model))
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for source, output in zip(videos, outputs):
        process_video(source, output, detect_model, segment_model, args, ffmpeg)


if __name__ == "__main__":
    main()
