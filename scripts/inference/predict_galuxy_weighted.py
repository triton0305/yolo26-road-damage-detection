import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "galuxy_finetune_v3_weighted" / "weights" / "best.pt"
SOURCE_DIR = Path(r"C:\Users\kccistc\Desktop\galuxy_ultra")
OUTPUT_DIR = Path(r"C:\Users\kccistc\Desktop\result_weighted")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    images = [
        path
        for path in SOURCE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")
    if not images:
        raise RuntimeError(f"입력 이미지가 없습니다: {SOURCE_DIR}")

    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source=str(SOURCE_DIR),
        imgsz=1024,
        conf=0.25,
        device=0,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(OUTPUT_DIR.parent),
        name=OUTPUT_DIR.name,
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
    print(f"저장 폴더: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
