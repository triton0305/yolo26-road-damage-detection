import argparse
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "galuxy_finetune_yolo26s" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Galaxy 비중 강화 데이터로 Detection v3를 추가 학습합니다.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "detect")
    parser.add_argument("--name", default="galuxy_finetune_v3_weighted")
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    data_yaml = args.data.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"1차 추가 학습 모델이 없습니다: {model_path}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"데이터 설정이 없습니다: {data_yaml}")

    model = YOLO(str(model_path))
    model.train(
        data=str(data_yaml),
        epochs=60,
        patience=15,
        imgsz=1024,
        batch=4,
        optimizer="AdamW",
        lr0=0.00005,
        lrf=0.2,
        warmup_epochs=0.0,
        warmup_bias_lr=0.00005,
        freeze=0,
        cls_pw=0.35,
        mosaic=0.15,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=10,
        workers=2,
        cache=False,
        device=args.device,
        amp=True,
        plots=True,
        save=True,
        save_period=5,
        project=str(args.project),
        name=args.name,
        seed=20260907,
        deterministic=True,
    )


if __name__ == "__main__":
    main()
