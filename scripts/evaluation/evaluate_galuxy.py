import argparse
import json
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
Box = tuple[int, float, float, float, float, float]


def read_yolo(path: Path) -> list[Box]:
    """YOLO 라벨을 읽는다. 예측 라벨의 confidence 열은 선택 사항이다."""
    if not path.is_file():
        return []

    boxes = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        values = line.split()
        if not values:
            continue
        if len(values) not in (5, 6):
            raise ValueError(f"잘못된 YOLO 라벨: {path}:{line_number}")

        class_id, x, y, width, height = map(float, values[:5])
        confidence = float(values[5]) if len(values) == 6 else 1.0
        boxes.append((int(class_id), x, y, width, height, confidence))
    return boxes


def intersection_over_union(first: Box, second: Box) -> float:
    """중심 좌표 형식의 두 YOLO 박스 사이 IoU를 계산한다."""
    _, first_x, first_y, first_width, first_height, _ = first
    _, second_x, second_y, second_width, second_height, _ = second

    first_x1 = first_x - first_width / 2
    first_y1 = first_y - first_height / 2
    first_x2 = first_x + first_width / 2
    first_y2 = first_y + first_height / 2
    second_x1 = second_x - second_width / 2
    second_y1 = second_y - second_height / 2
    second_x2 = second_x + second_width / 2
    second_y2 = second_y + second_height / 2

    intersection_width = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_height = max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection = intersection_width * intersection_height
    union = first_width * first_height + second_width * second_height - intersection
    return intersection / union if union > 0 else 0.0


def evaluate(
    images: Path, ground_truth: Path, predictions: Path, iou_threshold: float
) -> dict[str, int | float]:
    """같은 클래스의 박스를 confidence 순으로 1:1 매칭해 TP, FP, FN을 센다."""
    image_stems = sorted(
        {
            path.stem
            for path in images.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
    )
    if not image_stems:
        raise RuntimeError(f"평가 이미지가 없습니다: {images}")

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for image_stem in image_stems:
        truth_boxes = read_yolo(ground_truth / f"{image_stem}.txt")
        prediction_boxes = sorted(
            read_yolo(predictions / f"{image_stem}.txt"),
            key=lambda box: box[5],
            reverse=True,
        )
        unmatched_truth = set(range(len(truth_boxes)))

        for prediction in prediction_boxes:
            candidates = [
                (intersection_over_union(prediction, truth_boxes[index]), index)
                for index in unmatched_truth
                if prediction[0] == truth_boxes[index][0]
            ]
            best_iou, best_index = max(candidates, default=(0.0, -1))

            if best_iou >= iou_threshold:
                unmatched_truth.remove(best_index)
                true_positives += 1
            else:
                false_positives += 1

        false_negatives += len(unmatched_truth)

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "images": len(image_stems),
        "ground_truth_boxes": true_positives + false_negatives,
        "prediction_boxes": true_positives + false_positives,
        "iou_threshold": iou_threshold,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Galaxy 도로 손상 탐지 결과 평가")
    parser.add_argument("--images", type=Path, required=True, help="평가 이미지 폴더")
    parser.add_argument("--ground-truth", type=Path, required=True, help="정답 YOLO 라벨 폴더")
    parser.add_argument("--predictions", type=Path, required=True, help="예측 YOLO 라벨 폴더")
    parser.add_argument("--iou", type=float, default=0.5, help="TP 판정 IoU 기준")
    parser.add_argument("--output", type=Path, help="선택적 JSON 결과 저장 경로")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.images, args.ground_truth, args.predictions):
        if not path.is_dir():
            raise FileNotFoundError(f"폴더가 없습니다: {path}")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("IoU 기준은 0과 1 사이여야 합니다.")

    result = evaluate(args.images, args.ground_truth, args.predictions, args.iou)
    report = json.dumps(result, ensure_ascii=False, indent=2)
    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
