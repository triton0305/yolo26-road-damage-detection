import json
import random
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image


SEED = 20260907
VAL_RATIO = 0.20
CAPTURE_GROUP_GAP_SECONDS = 3
REHEARSAL_IMAGES_PER_CLASS = 20

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELING_ROOT = Path(r"C:\Users\kccistc\Desktop\galuxy_ultra_labeling")
SOURCE_IMAGES = LABELING_ROOT / "images"
SOURCE_LABELS = LABELING_ROOT / "labels_final_yolo"
EXIF_SOURCE_IMAGES = Path(r"C:\Users\kccistc\Desktop\galuxy_ultra")
OLD_TRAIN_IMAGES = Path(r"D:\road_yolo_detect\train\images")
OLD_TRAIN_LABELS = Path(r"D:\road_yolo_detect\train\labels")
DATASET_ROOT = Path(r"C:\Users\kccistc\Desktop\galuxy_ultra_finetune")

CLASS_NAMES = [
    "아스팔트 도로파임",
    "콘크리트 도로파임",
    "종방향균열",
    "횡방향균열",
    "거북등균열",
    "줄눈부파손",
    "십자파손",
    "절삭보수부파손",
    "긴급보수부파손",
    "규제봉",
    "맨홀",
    "배수로",
]
EMPTY_FEATURE = len(CLASS_NAMES)


def read_label(label_path: Path) -> tuple[set[int], Counter]:
    classes: set[int] = set()
    boxes: Counter = Counter()
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number}: YOLO 열이 5개가 아닙니다.")
        class_id = int(parts[0])
        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(f"{label_path}:{line_number}: 잘못된 클래스 {class_id}")
        coordinates = [float(value) for value in parts[1:]]
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError(f"{label_path}:{line_number}: 좌표 범위 오류")
        classes.add(class_id)
        boxes[class_id] += 1
    return classes, boxes


def capture_time(image_name: str) -> datetime | None:
    source = EXIF_SOURCE_IMAGES / image_name
    if not source.is_file():
        return None
    with Image.open(source) as image:
        value = image.getexif().get(306)
    if not value:
        return None
    return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")


def load_new_records() -> list[dict]:
    records = []
    for image_path in sorted(SOURCE_IMAGES.glob("*.jpg")):
        label_path = SOURCE_LABELS / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"라벨이 없습니다: {label_path}")
        classes, boxes = read_label(label_path)
        features = set(classes) if classes else {EMPTY_FEATURE}
        records.append(
            {
                "name": image_path.name,
                "image": image_path,
                "label": label_path,
                "classes": classes,
                "features": features,
                "boxes": boxes,
                "captured_at": capture_time(image_path.name),
            }
        )
    if len(records) != 184:
        raise RuntimeError(f"예상 이미지 184장, 실제 {len(records)}장")
    return records


def make_capture_groups(records: list[dict]) -> list[list[dict]]:
    dated = sorted(
        (record for record in records if record["captured_at"] is not None),
        key=lambda record: record["captured_at"],
    )
    undated = [record for record in records if record["captured_at"] is None]
    groups: list[list[dict]] = []
    for record in dated:
        if (
            not groups
            or (
                record["captured_at"] - groups[-1][-1]["captured_at"]
            ).total_seconds()
            > CAPTURE_GROUP_GAP_SECONDS
        ):
            groups.append([])
        groups[-1].append(record)
    groups.extend([[record] for record in undated])
    return groups


def choose_validation_groups(records: list[dict], groups: list[list[dict]]) -> set[str]:
    rng = random.Random(SEED)
    target_size = round(len(records) * VAL_RATIO)
    total_features = Counter(
        feature for record in records for feature in record["features"]
    )
    target_features = {
        feature: min(count - 1, max(1, round(count * VAL_RATIO)))
        if count >= 2
        else 0
        for feature, count in total_features.items()
    }

    group_stats = []
    for group in groups:
        feature_counts = Counter(
            feature for record in group for feature in record["features"]
        )
        group_stats.append((group, feature_counts))

    best_score = float("inf")
    best_selection: list[int] | None = None
    for _ in range(120_000):
        selected = [index for index in range(len(groups)) if rng.random() < VAL_RATIO]
        size = sum(len(groups[index]) for index in selected)
        feature_counts = Counter()
        for index in selected:
            feature_counts.update(group_stats[index][1])

        score = abs(size - target_size) * 12.0
        for feature, total in total_features.items():
            actual = feature_counts[feature]
            target = target_features[feature]
            score += abs(actual - target) / max(1, target)
            if total >= 2 and actual == 0:
                score += 25.0
            if total >= 2 and actual == total:
                score += 25.0
        if score < best_score:
            best_score = score
            best_selection = selected
            if size == target_size and score < 1.0:
                break

    if best_selection is None:
        raise RuntimeError("검증 세트를 선택하지 못했습니다.")
    validation_names = {
        record["name"]
        for index in best_selection
        for record in groups[index]
    }
    if not validation_names or len(validation_names) >= len(records):
        raise RuntimeError("잘못된 학습/검증 분할입니다.")
    return validation_names


def sample_old_rehearsal() -> tuple[list[Path], dict[int, int]]:
    rng = random.Random(SEED)
    samples: dict[int, list[Path]] = {
        class_id: [] for class_id in range(len(CLASS_NAMES))
    }
    label_paths = list(OLD_TRAIN_LABELS.glob("*.txt"))
    rng.shuffle(label_paths)

    def classes_only(label_path: Path) -> set[int]:
        classes = set()
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                classes.add(int(line.split(maxsplit=1)[0]))
        return classes

    batch_size = 4096
    with ThreadPoolExecutor(max_workers=32) as executor:
        for start in range(0, len(label_paths), batch_size):
            batch = label_paths[start : start + batch_size]
            for label_path, classes in zip(batch, executor.map(classes_only, batch)):
                for class_id in classes:
                    sample = samples[class_id]
                    if len(sample) < REHEARSAL_IMAGES_PER_CLASS:
                        sample.append(label_path)
            if all(
                len(sample) >= REHEARSAL_IMAGES_PER_CLASS
                for sample in samples.values()
            ):
                break

    missing = [
        class_id
        for class_id, sample in samples.items()
        if len(sample) < REHEARSAL_IMAGES_PER_CLASS
    ]
    if missing:
        raise RuntimeError(f"기존 데이터 리허설 표본 부족 클래스: {missing}")
    selected = sorted({path for paths in samples.values() for path in paths})
    coverage = Counter()
    for label_path in selected:
        classes, _ = read_label(label_path)
        coverage.update(classes)
    return selected, dict(coverage)


def copy_pair(image_path: Path, label_path: Path, split: str, prefix: str) -> None:
    image_output = DATASET_ROOT / split / "images" / f"{prefix}{image_path.name}"
    label_output = DATASET_ROOT / split / "labels" / f"{prefix}{label_path.name}"
    shutil.copy2(image_path, image_output)
    shutil.copy2(label_path, label_output)


def build_dataset() -> None:
    required = [
        SOURCE_IMAGES,
        SOURCE_LABELS,
        EXIF_SOURCE_IMAGES,
        OLD_TRAIN_IMAGES,
        OLD_TRAIN_LABELS,
    ]
    for path in required:
        if not path.is_dir():
            raise FileNotFoundError(f"필요한 폴더가 없습니다: {path}")
    if DATASET_ROOT.exists():
        raise FileExistsError(f"데이터셋 폴더가 이미 있습니다: {DATASET_ROOT}")

    records = load_new_records()
    groups = make_capture_groups(records)
    validation_names = choose_validation_groups(records, groups)
    train_records = [r for r in records if r["name"] not in validation_names]
    validation_records = [r for r in records if r["name"] in validation_names]
    rehearsal_labels, rehearsal_coverage = sample_old_rehearsal()

    for split in ("train", "valid"):
        (DATASET_ROOT / split / "images").mkdir(parents=True)
        (DATASET_ROOT / split / "labels").mkdir(parents=True)

    for record in train_records:
        copy_pair(record["image"], record["label"], "train", "galuxy_")
    for record in validation_records:
        copy_pair(record["image"], record["label"], "valid", "galuxy_")
    for label_path in rehearsal_labels:
        image_path = OLD_TRAIN_IMAGES / f"{label_path.stem}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(f"기존 이미지가 없습니다: {image_path}")
        copy_pair(image_path, label_path, "train", "rehearsal_")

    data = {
        "path": str(DATASET_ROOT),
        "train": "train/images",
        "val": "valid/images",
        "nc": len(CLASS_NAMES),
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    (DATASET_ROOT / "data.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    def feature_counts(items: list[dict]) -> dict[str, int]:
        counter = Counter(feature for item in items for feature in item["features"])
        return {
            ("empty" if key == EMPTY_FEATURE else str(key)): counter[key]
            for key in sorted(counter)
        }

    manifest = {
        "seed": SEED,
        "capture_group_gap_seconds": CAPTURE_GROUP_GAP_SECONDS,
        "new_train_images": len(train_records),
        "new_valid_images": len(validation_records),
        "rehearsal_train_images": len(rehearsal_labels),
        "new_train_feature_images": feature_counts(train_records),
        "new_valid_feature_images": feature_counts(validation_records),
        "rehearsal_class_image_coverage": rehearsal_coverage,
        "valid_files": sorted(record["name"] for record in validation_records),
        "rehearsal_files": [path.name for path in rehearsal_labels],
    }
    (DATASET_ROOT / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"데이터셋 생성 완료: {DATASET_ROOT}")


if __name__ == "__main__":
    build_dataset()
