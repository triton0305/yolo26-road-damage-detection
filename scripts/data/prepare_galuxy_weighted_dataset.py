import argparse
from pathlib import Path

import prepare_galuxy_finetune as preparation


# 직접 라벨링한 데이터가 학습셋의 대부분이 되도록 기존 데이터는
# 클래스별 최소 리허설 표본만 남긴다. 검증 37장은 기존 분할을 그대로 쓴다.
preparation.REHEARSAL_IMAGES_PER_CLASS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Galaxy 비중을 높인 Detection 데이터셋을 구성합니다.")
    parser.add_argument("--labeling-root", type=Path, default=preparation.LABELING_ROOT)
    parser.add_argument("--exif-source", type=Path, default=preparation.EXIF_SOURCE_IMAGES)
    parser.add_argument("--old-dataset", type=Path, default=preparation.OLD_DATASET_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=preparation.PROJECT_ROOT / "datasets" / "galuxy_ultra_finetune_v2",
    )
    return parser.parse_args()


if __name__ == "__main__":
    preparation.configure_paths(parse_args())
    preparation.build_dataset()
