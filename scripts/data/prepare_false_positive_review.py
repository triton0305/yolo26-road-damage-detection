import argparse
import csv
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELING_ROOT = PROJECT_ROOT / "datasets" / "galuxy_ultra_labeling"
PREDICTION_ROOT = PROJECT_ROOT / "outputs" / "result_weighted"
OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "galuxy_false_positive_review"
IOU_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detection 오탐 후보를 검수 폴더로 모읍니다.")
    parser.add_argument("--labeling-root", type=Path, default=LABELING_ROOT)
    parser.add_argument("--prediction-root", type=Path, default=PREDICTION_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def read_yolo(path: Path) -> list[tuple[int, float, float, float, float, float]]:
    if not path.is_file():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        class_id, x, y, width, height = map(float, values[:5])
        confidence = float(values[5]) if len(values) >= 6 else 1.0
        boxes.append((int(class_id), x, y, width, height, confidence))
    return boxes


def iou(a: tuple, b: tuple) -> float:
    _, ax, ay, aw, ah, _ = a
    _, bx, by, bw, bh, _ = b
    ax1, ay1, ax2, ay2 = ax - aw / 2, ay - ah / 2, ax + aw / 2, ay + ah / 2
    bx1, by1, bx2, by2 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def false_predictions(image_stem: str) -> list[tuple]:
    truth = read_yolo(LABELING_ROOT / "labels_final_yolo" / f"{image_stem}.txt")
    predictions = sorted(
        read_yolo(PREDICTION_ROOT / "labels" / f"{image_stem}.txt"),
        key=lambda box: box[5],
        reverse=True,
    )
    unmatched_truth = set(range(len(truth)))
    false_boxes = []
    for prediction in predictions:
        candidates = [
            (iou(prediction, truth[index]), index)
            for index in unmatched_truth
            if prediction[0] == truth[index][0]
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= IOU_THRESHOLD:
            unmatched_truth.remove(best_index)
        else:
            false_boxes.append(prediction)
    return false_boxes


def main() -> None:
    global LABELING_ROOT, PREDICTION_ROOT, OUTPUT_ROOT
    args = parse_args()
    LABELING_ROOT = args.labeling_root.expanduser().resolve()
    PREDICTION_ROOT = args.prediction_root.expanduser().resolve()
    OUTPUT_ROOT = args.output.expanduser().resolve()

    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"검수 폴더가 이미 있습니다: {OUTPUT_ROOT}")

    classes = (LABELING_ROOT / "classes.txt").read_text(encoding="utf-8").splitlines()
    rows = []
    candidates = []
    for image_path in sorted((LABELING_ROOT / "images").glob("*.jpg")):
        false_boxes = false_predictions(image_path.stem)
        if false_boxes:
            candidates.append((image_path, false_boxes))

    for folder in ("images", "annotations_xlabel", "labels_current_yolo", "reference_predictions"):
        (OUTPUT_ROOT / folder).mkdir(parents=True, exist_ok=True)

    shutil.copy2(LABELING_ROOT / "classes.txt", OUTPUT_ROOT / "classes.txt")
    for image_path, boxes in candidates:
        stem = image_path.stem
        shutil.copy2(image_path, OUTPUT_ROOT / "images" / image_path.name)
        shutil.copy2(
            LABELING_ROOT / "annotations_xlabel" / f"{stem}.json",
            OUTPUT_ROOT / "annotations_xlabel" / f"{stem}.json",
        )
        shutil.copy2(
            LABELING_ROOT / "labels_final_yolo" / f"{stem}.txt",
            OUTPUT_ROOT / "labels_current_yolo" / f"{stem}.txt",
        )
        shutil.copy2(
            PREDICTION_ROOT / image_path.name,
            OUTPUT_ROOT / "reference_predictions" / image_path.name,
        )
        for class_id, x, y, width, height, confidence in boxes:
            rows.append(
                {
                    "image": image_path.name,
                    "class_id": class_id,
                    "class_name": classes[class_id],
                    "confidence": f"{confidence:.4f}",
                    "x_center": f"{x:.6f}",
                    "y_center": f"{y:.6f}",
                    "width": f"{width:.6f}",
                    "height": f"{height:.6f}",
                }
            )

    with (OUTPUT_ROOT / "false_positive_report.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as report:
        writer = csv.DictWriter(report, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    launcher = """@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "WORK=%~dp0"
xanylabeling --filename "%WORK%images" --output "%WORK%annotations_xlabel" --labels "%WORK%classes.txt" --validatelabel exact --autosave --nodata --nosortlabels --no-auto-update-check
if errorlevel 1 pause
"""
    (OUTPUT_ROOT / "오탐_검수_시작.cmd").write_text(launcher, encoding="utf-8")

    guide = """오탐 검수 방법

1. reference_predictions에서 모델이 잘못 그린 박스를 확인한다.
2. 오탐_검수_시작.cmd를 실행하면 같은 원본과 기존 수동 라벨이 열린다.
3. 배경을 잘못 잡은 순수 오탐이면 기존 라벨을 추가하거나 지우지 않는다.
4. 실제 대상인데 클래스/박스가 잘못됐다면 기존 라벨을 수정한다.
5. 미탐된 실제 대상이 보이면 새 박스를 추가한다.

주의: reference_predictions 이미지는 박스가 픽셀에 그려진 참고용이므로 직접 라벨링하지 않는다.
"""
    (OUTPUT_ROOT / "검수방법.txt").write_text(guide, encoding="utf-8")
    print(f"오탐 후보 이미지: {len(candidates)}장")
    print(f"오탐 후보 박스: {len(rows)}개")
    print(f"검수 폴더: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
