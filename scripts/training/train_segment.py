import os
from pathlib import Path

# OpenMP 중복 로드 충돌 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def train_segmentation():
    data_yaml = PROJECT_ROOT / "configs" / "data_segment.yaml"
    output_root = PROJECT_ROOT / "runs" / "segment"
    run_name = "road_segment_yolo26s"
    last_checkpoint = output_root / run_name / "weights" / "last.pt"

    # 기존 체크포인트가 있으면 처음부터 학습하지 않고 자동으로 이어서 실행
    if last_checkpoint.exists():
        model = YOLO(str(last_checkpoint))
        return model.train(
            resume=True,
            workers=2,
            device=0,
            save_period=1,
        )

    model = YOLO("yolo26s-seg.pt")

    return model.train(
        data=str(data_yaml),
        epochs=20,
        imgsz=1024,
        batch=1,
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
    train_segmentation()
