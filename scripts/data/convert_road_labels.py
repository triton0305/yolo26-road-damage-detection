import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_JSON_ROOT = Path(
    r"D:\road\060.고해상도 도로노면 이미지 데이터"
    r"\3.개방데이터\1.데이터"
)
DEFAULT_IMAGE_ROOT = Path(r"D:\road")
DEFAULT_DETECT_ROOT = Path(r"D:\road_yolo_detect")
DEFAULT_SEGMENT_ROOT = Path(r"D:\road_yolo_segment")

# AI Hub category_id(1부터 시작) -> 새 YOLO class id(0부터 시작)
DETECT_CLASS_MAP = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    16: 9,
    17: 10,
    18: 11,
}

SEGMENT_CLASS_MAP = {
    10: 0,
    11: 1,
    12: 2,
    13: 3,
    14: 4,
    15: 5,
}


@dataclass
class SplitStats:
    json_files: int = 0
    detection_images: int = 0
    segmentation_images: int = 0
    detection_objects: int = 0
    segmentation_objects: int = 0
    missing_images: int = 0
    invalid_annotations: int = 0


def clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def convert_bbox(
    category_id: int,
    bbox: list[float],
    width: int,
    height: int,
) -> str | None:
    """AI Hub [x, y, width, height] bbox를 YOLO detection 형식으로 변환한다."""
    class_id = DETECT_CLASS_MAP.get(category_id)
    if class_id is None or len(bbox) != 4 or width <= 0 or height <= 0:
        return None

    x, y, box_width, box_height = map(float, bbox)
    if box_width <= 0 or box_height <= 0:
        return None

    x_center = clamp((x + box_width / 2) / width)
    y_center = clamp((y + box_height / 2) / height)
    normalized_width = clamp(box_width / width)
    normalized_height = clamp(box_height / height)

    if normalized_width == 0 or normalized_height == 0:
        return None

    return (
        f"{class_id} {x_center:.6f} {y_center:.6f} "
        f"{normalized_width:.6f} {normalized_height:.6f}"
    )


def iter_polygons(segmentation: object) -> Iterable[list[float]]:
    """flat polygon과 [[polygon], ...] 형식을 모두 처리한다."""
    if not isinstance(segmentation, list) or not segmentation:
        return

    if all(isinstance(value, (int, float)) for value in segmentation):
        yield [float(value) for value in segmentation]
        return

    for polygon in segmentation:
        if not isinstance(polygon, list) or not polygon:
            continue

        if all(isinstance(value, (int, float)) for value in polygon):
            yield [float(value) for value in polygon]
            continue

        # [[x, y], [x, y], ...] 형식도 허용한다.
        if all(
            isinstance(point, list)
            and len(point) == 2
            and all(isinstance(value, (int, float)) for value in point)
            for point in polygon
        ):
            yield [float(value) for point in polygon for value in point]


def convert_polygon(
    category_id: int,
    segmentation: object,
    width: int,
    height: int,
) -> list[str]:
    """AI Hub polygon을 YOLO segmentation 형식의 한 줄 이상으로 변환한다."""
    class_id = SEGMENT_CLASS_MAP.get(category_id)
    if class_id is None or width <= 0 or height <= 0:
        return []

    lines: list[str] = []
    for polygon in iter_polygons(segmentation):
        if len(polygon) < 6 or len(polygon) % 2 != 0:
            continue

        normalized: list[str] = []
        for index in range(0, len(polygon), 2):
            normalized.append(f"{clamp(polygon[index] / width):.6f}")
            normalized.append(f"{clamp(polygon[index + 1] / height):.6f}")

        lines.append(f"{class_id} {' '.join(normalized)}")

    return lines


def convert_annotation(data: dict) -> tuple[str, list[str], list[str], int]:
    """JSON 한 개를 detection/segmentation 라벨 줄로 나눈다."""
    image_info = data.get("images", {})
    file_name = image_info.get("file_name", "")
    width = int(image_info.get("width", 0))
    height = int(image_info.get("height", 0))

    detection_lines: list[str] = []
    segmentation_lines: list[str] = []
    invalid_annotations = 0

    for annotation in data.get("annotations", []):
        category_id = int(annotation.get("category_id", -1))

        if category_id in DETECT_CLASS_MAP:
            line = convert_bbox(
                category_id,
                annotation.get("bbox", []),
                width,
                height,
            )
            if line is None:
                invalid_annotations += 1
            else:
                detection_lines.append(line)

        if category_id in SEGMENT_CLASS_MAP:
            lines = convert_polygon(
                category_id,
                annotation.get("segmentation", []),
                width,
                height,
            )
            if not lines:
                invalid_annotations += 1
            else:
                segmentation_lines.extend(lines)

    return file_name, detection_lines, segmentation_lines, invalid_annotations


def link_or_copy_image(source: Path, destination: Path, copy_images: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    if copy_images:
        shutil.copy2(source, destination)
    else:
        try:
            os.link(source, destination)
        except OSError as error:
            raise OSError(
                f"하드링크 생성 실패: {source} -> {destination}\n"
                "두 경로를 같은 드라이브에 두거나 --copy-images 옵션을 사용하세요."
            ) from error


def write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_task_sample(
    source_image: Path,
    output_root: Path,
    split: str,
    label_lines: list[str],
    copy_images: bool,
) -> None:
    output_image = output_root / split / "images" / source_image.name
    output_label = output_root / split / "labels" / f"{source_image.stem}.txt"
    link_or_copy_image(source_image, output_image, copy_images)
    write_label(output_label, label_lines)


def convert_dataset(
    json_root: Path,
    image_root: Path,
    detect_root: Path,
    segment_root: Path,
    limit: int,
    copy_images: bool,
) -> None:
    """Training/Validation JSON 전체를 두 개의 YOLO 데이터셋으로 변환한다."""
    split_settings = (
        ("Training", "train"),
        ("Validation", "valid"),
    )

    for source_split, output_split in split_settings:
        label_root = json_root / source_split / "02.라벨링데이터"
        source_image_root = image_root / output_split / "images"
        json_files = label_root.rglob("*.json")
        stats = SplitStats()

        for json_path in json_files:
            if limit and stats.json_files >= limit:
                break

            stats.json_files += 1
            try:
                data = json.loads(json_path.read_text(encoding="utf-8-sig"))
                file_name, detect_lines, segment_lines, invalid = convert_annotation(data)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                stats.invalid_annotations += 1
                print(f"[JSON 오류] {json_path}: {error}")
                continue

            stats.invalid_annotations += invalid
            source_image = source_image_root / file_name
            if not file_name or not source_image.is_file():
                stats.missing_images += 1
                continue

            if detect_lines:
                prepare_task_sample(
                    source_image,
                    detect_root,
                    output_split,
                    detect_lines,
                    copy_images,
                )
                stats.detection_images += 1
                stats.detection_objects += len(detect_lines)

            if segment_lines:
                prepare_task_sample(
                    source_image,
                    segment_root,
                    output_split,
                    segment_lines,
                    copy_images,
                )
                stats.segmentation_images += 1
                stats.segmentation_objects += len(segment_lines)

            if stats.json_files % 5000 == 0:
                print(
                    f"[{output_split}] {stats.json_files:,}개 처리 | "
                    f"detect {stats.detection_images:,}장 | "
                    f"segment {stats.segmentation_images:,}장"
                )

        print(f"\n[{output_split} 완료]")
        print(f"JSON: {stats.json_files:,}")
        print(
            f"Detection: {stats.detection_images:,}장 / "
            f"{stats.detection_objects:,}개 객체"
        )
        print(
            f"Segmentation: {stats.segmentation_images:,}장 / "
            f"{stats.segmentation_objects:,}개 객체"
        )
        print(f"누락 이미지: {stats.missing_images:,}")
        print(f"잘못된 어노테이션: {stats.invalid_annotations:,}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Hub 도로노면 JSON을 YOLO detection/segmentation 데이터로 분리합니다."
    )
    parser.add_argument("--json-root", type=Path, default=DEFAULT_JSON_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--detect-root", type=Path, default=DEFAULT_DETECT_ROOT)
    parser.add_argument("--segment-root", type=Path, default=DEFAULT_SEGMENT_ROOT)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="split마다 처리할 JSON 수. 0이면 전체를 처리합니다.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="하드링크 대신 이미지를 복사합니다. 기본값은 디스크를 절약하는 하드링크입니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.json_root.is_dir():
        raise FileNotFoundError(f"JSON 루트가 없습니다: {args.json_root}")
    if not args.image_root.is_dir():
        raise FileNotFoundError(f"이미지 루트가 없습니다: {args.image_root}")
    if args.limit < 0:
        raise ValueError("--limit는 0 이상이어야 합니다.")

    convert_dataset(
        json_root=args.json_root,
        image_root=args.image_root,
        detect_root=args.detect_root,
        segment_root=args.segment_root,
        limit=args.limit,
        copy_images=args.copy_images,
    )


if __name__ == "__main__":
    main()
