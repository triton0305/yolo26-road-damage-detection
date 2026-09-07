import os
from pathlib import Path

# OpenMP 중복 로드 충돌 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def train_detection():
    data_yaml = PROJECT_ROOT / "configs" / "data_detect.yaml"
    output_root = PROJECT_ROOT / "runs" / "detect"
    run_name = "detect_20epochs"
    total_epochs = 20
    last_checkpoint = output_root / run_name / "weights" / "last.pt"

    # last.pt의 저장 epoch 다음부터 학습을 재개한다.
    if last_checkpoint.exists():
        model = YOLO(str(last_checkpoint))
        completed_epochs = int(model.ckpt.get("epoch", -1)) + 1
        next_epoch = completed_epochs + 1
        print(f"{completed_epochs} epochs 완료: {next_epoch}/{total_epochs} epoch부터 재개")
        return model.train(
            resume=True,
            workers=2,
            device=0,
            save_period=1,
        )

    local_model = PROJECT_ROOT / "models" / "pretrained" / "yolo26s.pt"
    model = YOLO(str(local_model) if local_model.is_file() else "yolo26s.pt")

    return model.train(
        data=str(data_yaml),
        epochs=total_epochs,
        imgsz=1024,
        batch=-1,
        optimizer="auto",
        patience=10,
        workers=2,
        cache=False,
        device=0,
        amp=True,
        plots=True,
        save=True,
        save_period=1,
        project=str(output_root),
        name=run_name,
    )


if __name__ == "__main__":
    train_detection()
