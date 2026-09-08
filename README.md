# YOLO26 도로 손상 탐지

AI Hub의 고해상도 도로 노면 데이터와 직접 촬영한 Galaxy 이미지를 이용해 포트홀과 균열 등을 탐지하는 프로젝트입니다. Ultralytics YOLO26s를 기반으로 데이터 변환, 추가 학습, 추론, 오탐 검수와 재평가 과정을 구성했습니다.

## 최종 모델

- 모델: YOLO26s
- 학습 실행: `runs/detect/galuxy_finetune_v4_revised`
- 가중치: `runs/detect/galuxy_finetune_v4_revised/weights/best.pt`
- 평가 데이터: 직접 촬영 이미지 184장, 정답 박스 163개
- TP / FP / FN: 92 / 24 / 71
- Precision / Recall / F1: 0.793 / 0.564 / 0.659

모델 가중치와 학습 결과는 용량 문제로 Git에 포함하지 않습니다. 실험별 변경점과 평가 기준은 [`results/EXPERIMENTS.md`](results/EXPERIMENTS.md)에 정리되어 있습니다.

## 🎥 Real-World Inference Demo

[![YOLO Road Surface Detection Demo](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)

> AI Hub 고해상도 도로노면 이미지 데이터로 학습한 모델의 실도로 환경 추론 결과입니다.

## 프로젝트 흐름

```text
AI Hub 원본 JSON·이미지
  → YOLO 탐지·세그멘테이션 데이터 변환
  → 기본 모델 학습
  → 직접 촬영 Galaxy 데이터 추가
  → v3 weighted 학습
  → 오탐 검수와 라벨 수정
  → v4 revised 추가 학습
  → 동일 데이터 기준 TP·FP·FN 평가
```

## 프로젝트 구조

```text
configs/              데이터셋 YAML 설정
scripts/data/         데이터 변환과 학습 데이터 준비
scripts/training/     기본 학습과 Galaxy 미세조정
scripts/inference/    모델 추론
scripts/evaluation/   TP·FP·FN 기반 평가
results/              실험 기록 문서
models/pretrained/    로컬 사전학습 가중치, Git 제외
runs/                 학습 체크포인트와 그래프, Git 제외
```

## 주요 실행 파일

| 파일 | 역할 |
|---|---|
| `scripts/data/convert_road_labels.py` | AI Hub 라벨을 YOLO 형식으로 변환 |
| `scripts/data/prepare_galuxy_weighted_dataset.py` | Galaxy 비중을 높인 v3 데이터셋 준비 |
| `scripts/training/train_galuxy_weighted.py` | v3 weighted 모델 학습 |
| `scripts/training/train_galuxy_revised.py` | 수정 라벨로 최종 v4 모델 학습 |
| `scripts/inference/predict_galuxy_weighted.py` | 이미지 일괄 추론과 예측 라벨 저장 |
| `scripts/evaluation/evaluate_galuxy.py` | 같은 클래스·IoU 0.5 기준 성능 평가 |

## 개발 환경

- Python 3.11.15
- Ultralytics YOLO11 / YOLO26s
- PyTorch
- OpenCV
- Pillow
- PyYAML

## 데이터 출처

- 데이터셋: AI Hub 고해상도 도로 노면 데이터
- 수행기관: 에이치씨아이플러스(주)
- 제공기관: 과학기술정보통신부 / 한국지능정보사회진흥원(NIA)
- 링크: <https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71781>
