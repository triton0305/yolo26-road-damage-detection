# 프로젝트 파일 구분

## GitHub에 포함할 최종 파일

| 구분 | 파일 | 용도 |
|---|---|---|
| 문서 | `README.md` | 프로젝트 개요, 결과, 실행 방법 |
| 문서 | `results/EXPERIMENTS.md` | Detection·Segmentation 실험 기록 |
| 결과 | `results/detection/` | Detection 4개 실험의 학습 곡선과 지표 CSV |
| 결과 | `results/segmentation/` | Segmentation 3개 실험의 학습 곡선과 지표 CSV |
| 환경 | `requirements.txt` | Python 의존성 |
| 설정 | `configs/data_detect.yaml` | AI Hub Detection 데이터 설정 |
| 설정 | `configs/data_segment.yaml` | AI Hub Segmentation 데이터 설정 |
| 설정 | `configs/data_segment_manual.example.yaml` | 수동 Segmentation 데이터 예시 설정 |
| 데이터 | `scripts/data/convert_road_labels.py` | AI Hub 라벨 YOLO 형식 변환 |
| 데이터 | `scripts/data/prepare_galuxy_finetune.py` | Galaxy Detection 데이터 준비 공통 로직 |
| 데이터 | `scripts/data/prepare_galuxy_weighted_dataset.py` | Detection v3 weighted 데이터 구성 |
| 데이터 | `scripts/data/prepare_false_positive_review.py` | Detection 오탐 검수 데이터 준비 |
| 학습 | `scripts/training/train_detect.py` | 기본 Detection 학습 |
| 학습 | `scripts/training/train_segment.py` | 기본 Segmentation 학습 |
| 학습 | `scripts/training/train_galuxy_weighted.py` | Detection v3 weighted 학습 |
| 학습 | `scripts/training/train_galuxy_revised.py` | Detection v4 revised 학습 |
| 학습 | `scripts/training/train_galuxy_segment_manual.py` | 수동 폴리곤 Segmentation 추가 학습 |
| 추론 | `scripts/inference/predict_galuxy_weighted.py` | v3 비교용 이미지 추론 |
| 추론 | `scripts/inference/predict_combined_final.py` | 최종 이미지 Detection+Segmentation 통합 |
| 추론 | `scripts/inference/predict_combined_video.py` | 최종 영상 통합 및 1080p30 인코딩 |
| 평가 | `scripts/evaluation/evaluate_galuxy.py` | Detection TP·FP·FN 평가 |

## 최종 사용 모델

가중치 파일은 모두 `runs/` 아래에 있으며 Git에는 포함하지 않는다.

| 작업 | 로컬 가중치 | 용도 |
|---|---|---|
| Detection | `runs/detect/galuxy_finetune_v4_revised/weights/best.pt` | 최종 도로 손상 탐지 |
| Segmentation 기반 | `runs/segment/road_segment_yolo26s/weights/best.pt` | AI Hub 5만 장 학습 모델 |
| Segmentation 통합 | `runs/segment/galuxy_segment_manual_v1/weights/epoch30.pt` | 수동 라벨 영향 중간 체크포인트 |

## GitHub에 포함하지 않는 항목

- `runs/`: 체크포인트와 전체 학습 산출물 (`results/segmentation/`에 선별 복사한 자료만 공개)
- `models/`, `*.pt`: 모델 가중치
- 원본 AI Hub 데이터와 직접 촬영 이미지·영상
- 수동 라벨 데이터와 생성된 YOLO 라벨
- `result_*`, `outputs/`: 이미지·영상 추론 결과
- `.venv/`, `__pycache__/`, `*.cache`: 실행 환경과 캐시
- `scripts/legacy/`: 이전 YOLO11 참고 코드
- `scripts/inference/predict_segment_tagged.py`: 최종 통합 전 중간 테스트 도구

위 파일은 삭제하지 않고 `.gitignore`로만 제외한다.

## 로컬 최종 결과

| 위치 | 내용 |
|---|---|
| `<로컬 출력 경로>/final_result` | 원본 이미지 177장의 통합 결과 |
| `<로컬 출력 경로>/movie_result` | YouTube용 1080p30 통합 영상 2개 |

## 최종 Detection 평가

- 평가셋: 직접 촬영 이미지 184장, 수정 정답 박스 163개
- TP / FP / FN: 92 / 24 / 71
- Precision / Recall / F1: 0.793 / 0.564 / 0.659

평가 기준과 실험별 차이는 `results/EXPERIMENTS.md`에 기록한다.
