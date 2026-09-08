"""Fine-tune an AI Hub segmentation checkpoint with manually labeled images."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--project",
        type=Path,
        default=PROJECT_ROOT / "runs" / "segment",
    )
    parser.add_argument("--name", default="galuxy_segment_manual_v1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr0", type=float, default=0.00005)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"AI Hub checkpoint not found: {args.model}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset config not found: {args.data}")

    model = YOLO(str(args.model))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.2,
        warmup_epochs=0.0,
        freeze=args.freeze,
        mosaic=0.10,
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
        seed=20260908,
        deterministic=True,
    )


if __name__ == "__main__":
    main()
