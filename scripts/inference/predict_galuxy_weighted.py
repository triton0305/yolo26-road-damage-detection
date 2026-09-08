import argparse
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "galuxy_finetune_v3_weighted" / "weights" / "best.pt"
SOURCE_DIR = PROJECT_ROOT / "datasets" / "galuxy_ultra"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "result_weighted"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detection v3 weighted 모델로 이미지를 추론합니다.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    source_dir = args.source.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    images = [
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not model_path.is_file():
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
    if not images:
        raise RuntimeError(f"입력 이미지가 없습니다: {source_dir}")

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(source_dir),
        imgsz=1024,
        conf=0.25,
        device=args.device,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        stream=True,
        verbose=False,
    )

    processed = detected_images = detections = 0
    for result in results:
        processed += 1
        count = 0 if result.boxes is None else len(result.boxes)
        if count:
            detected_images += 1
            detections += count

    print(f"처리 완료: {processed}/{len(images)}장")
    print(f"검출된 사진: {detected_images}장")
    print(f"검출 객체: {detections}개")
    print(f"저장 폴더: {output_dir}")


if __name__ == "__main__":
    main()
