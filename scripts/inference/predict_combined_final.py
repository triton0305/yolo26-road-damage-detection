"""Combine segmentation masks/boxes and detection boxes on the same images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    images = sorted(
        path for path in args.source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError(f"No images found: {args.source}")

    args.output.mkdir(parents=True)
    detect_labels = args.output / "labels_detect"
    segment_labels = args.output / "labels_segment"
    detect_labels.mkdir()
    segment_labels.mkdir()

    detect_model = YOLO(str(args.detect_model))
    segment_model = YOLO(str(args.segment_model))
    segment_results = segment_model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.segment_conf,
        classes=None,
        device=args.device,
        retina_masks=True,
        stream=True,
        save=False,
        verbose=False,
    )
    detect_results = detect_model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.detect_conf,
        classes=None,
        device=args.device,
        stream=True,
        save=False,
        verbose=False,
    )

    processed = segment_count = detect_count = 0
    for processed, (segment_result, detect_result) in enumerate(
        zip(segment_results, detect_results), start=1
    ):
        segment_name = Path(segment_result.path).name
        detect_name = Path(detect_result.path).name
        if segment_name != detect_name:
            raise RuntimeError(f"Result order mismatch: {segment_name} != {detect_name}")

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
        cv2.imwrite(
            str(args.output / segment_name),
            combined,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        stem = Path(segment_name).stem
        if segment_result.boxes is not None and len(segment_result.boxes):
            segment_result.save_txt(str(segment_labels / f"{stem}.txt"), save_conf=True)
            segment_count += len(segment_result.boxes)
        if detect_result.boxes is not None and len(detect_result.boxes):
            detect_result.save_txt(str(detect_labels / f"{stem}.txt"), save_conf=True)
            detect_count += len(detect_result.boxes)

        if processed % 10 == 0:
            print(f"processed={processed}/{len(images)}", flush=True)

    print(
        f"completed={processed} segment={segment_count} detect={detect_count} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
