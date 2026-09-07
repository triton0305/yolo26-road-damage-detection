import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "galuxy_finetune_yolo26s" / "weights" / "best.pt"
DATA_YAML = Path(r"C:\Users\kccistc\Desktop\galuxy_ultra_finetune_v2\data.yaml")


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"1차 추가 학습 모델이 없습니다: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    model.train(
        data=str(DATA_YAML),
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
        device=0,
        amp=True,
        plots=True,
        save=True,
        save_period=5,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="galuxy_finetune_v3_weighted",
        seed=20260907,
        deterministic=True,
    )


if __name__ == "__main__":
    main()
