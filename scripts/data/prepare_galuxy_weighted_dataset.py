from pathlib import Path

import prepare_galuxy_finetune as preparation


# 직접 라벨링한 데이터가 학습셋의 대부분이 되도록 기존 데이터는
# 클래스별 최소 리허설 표본만 남긴다. 검증 37장은 기존 분할을 그대로 쓴다.
preparation.REHEARSAL_IMAGES_PER_CLASS = 5
preparation.DATASET_ROOT = Path(r"C:\Users\kccistc\Desktop\galuxy_ultra_finetune_v2")


if __name__ == "__main__":
    preparation.build_dataset()
