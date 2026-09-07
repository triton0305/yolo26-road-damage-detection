# 프로젝트 파일 구분

## GitHub에 올릴 현재 코드

| 파일 | 용도 |
|---|---|
| `scripts/data/convert_road_labels.py` | 원본 도로 라벨을 YOLO 형식으로 변환 |
| `scripts/data/prepare_galuxy_finetune.py` | Galaxy 데이터 분할·복사 공통 로직 |
| `scripts/data/prepare_galuxy_weighted_dataset.py` | 직접 라벨 데이터 73.1% 비중의 최종 데이터셋 준비 |
| `scripts/data/prepare_false_positive_review.py` | 최종 예측에서 오탐 후보만 검수 폴더로 분리 |
| `scripts/training/train_detect.py` | 기본 탐지 모델 학습 |
| `scripts/training/train_segment.py` | 기본 세그멘테이션 모델 학습 |
| `scripts/training/train_galuxy_weighted.py` | v3 가중 재학습 실행 |
| `scripts/training/train_galuxy_revised.py` | 현재 최종 v4 수정 데이터 재학습 실행 |
| `scripts/inference/predict_galuxy_weighted.py` | v3 모델로 이미지 추론 |
| `scripts/evaluation/evaluate_galuxy.py` | 동일 클래스·IoU 0.5 기준 TP·FP·FN 평가 |
| `results/EXPERIMENTS.md` | 실험별 변경점과 평가 결과 기록 |
| `configs/data_detect.yaml` | 기본 탐지 데이터 설정 |
| `configs/data_segment.yaml` | 기본 세그멘테이션 데이터 설정 |

## 디렉터리 구조

```text
project/
├─ configs/             # 데이터셋 YAML
├─ models/pretrained/   # 로컬 사전학습 가중치, Git 제외
├─ scripts/
│  ├─ data/             # 데이터 변환과 준비
│  ├─ training/         # 탐지·세그멘테이션 학습
│  ├─ inference/        # 최종 모델 추론
│  ├─ evaluation/       # TP·FP·FN 기반 모델 평가
│  └─ legacy/           # 이전 YOLO11 참고 코드
├─ results/             # Git에는 실험 기록 문서만 포함
└─ runs/
   ├─ detect/           # 탐지 모델과 결과
   └─ segment/          # 세그멘테이션 모델과 결과
```

## 로컬에서 사용하는 현재 모델 및 결과

`runs/`는 Git에서 제외되므로 GitHub에는 올라가지 않는다.

| 폴더 | 상태 | 설명 |
|---|---|---|
| `runs/detect/detect_20epochs` | 기반 | 20에포크 도로 탐지 모델 |
| `runs/detect/galuxy_finetune_yolo26s` | 중간 | 1차 셀프 라벨 미세조정 모델 |
| `runs/detect/galuxy_finetune_v3_weighted` | 기반 | 직접 라벨 비중을 높인 v3 모델 |
| `runs/detect/galuxy_finetune_v4_revised` | **최종 사용** | 수정 라벨로 추가 학습한 최종 모델 |
| `runs/segment/road_segment_yolo26s` | **보존** | 도로 세그멘테이션 학습 모델 |
| `runs/segment/galaxy_segment_test` | **보존** | 세그멘테이션 테스트 결과 |
| `runs/segment/galaxy_segment_test_conf025` | **보존** | confidence 0.25 세그멘테이션 테스트 결과 |

실제 사용하는 가중치는 `runs/detect/galuxy_finetune_v4_revised/weights/best.pt`이다.

## GitHub에 올리지 않는 파일

- `runs/`: 학습 체크포인트와 그래프
- `*.pt`, `*.pth`: 모델 가중치
- `__pycache__/`, `*.cache`: Python 및 데이터 캐시
- 원본 학습 데이터와 직접 촬영 이미지
- 추론 결과 이미지

## 로컬 보관 폴더

현재 사용하지 않는 실행 결과와 이전 시도 스크립트는 삭제하지 않고 다음 위치로 이동했다.

`C:\Users\kccistc\Desktop\galuxy_project_archive_20260907`

최종 결과 이미지와 평가 원본은 로컬 결과 폴더에 보관한다.

## 최종 모델 검증 결과

- 평가셋: 직접 촬영 이미지 184장, 수정 정답 박스 163개
- TP / FP / FN: 92 / 24 / 71
- Precision: 0.793
- Recall: 0.564
- F1: 0.659

평가 기준과 실험별 비교는 `results/EXPERIMENTS.md`에 기록되어 있다.
