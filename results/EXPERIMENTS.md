# 실험 기록

## 최종 평가 기준

- 평가 데이터: 직접 촬영한 Galaxy 이미지 184장
- 수정 정답 라벨: 163개 박스
- 예측 confidence 기준: 0.25
- TP 판정: 예측과 정답의 클래스가 같고 IoU가 0.50 이상
- 매칭 방식: confidence가 높은 예측부터 정답과 1:1 매칭
- 평가 코드: `scripts/evaluation/evaluate_galuxy.py`

## 실험 비교

| 실험 | 주요 변경 | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| v3 weighted | 직접 라벨 데이터 비중 73.1%, 클래스별 기존 데이터 5장 유지 | 64 | 29 | 99 | 0.688 | 0.393 | 0.500 |
| v4 revised | v3 최적 가중치에서 수정 라벨로 추가 학습, 수정 표본 반복, mosaic 0.10 | 92 | 24 | 71 | 0.793 | 0.564 | 0.659 |

두 모델은 동일한 184장과 수정 정답 라벨 163개를 기준으로 비교했다. v4 revised는 v3 weighted보다 TP가 28개 증가하고 FP가 5개, FN이 28개 감소했다. F1은 0.500에서 0.659로 0.159 증가했다.

## 실험별 역할

### v3 weighted

- 직접 촬영한 Galaxy 데이터가 학습 데이터의 73.1%가 되도록 비중을 높였다.
- 기존 도로 데이터는 클래스별 5장을 유지해 이전 학습 내용을 일부 보존했다.
- 오탐 검수와 라벨 수정의 기준 모델로 사용했다.

### v4 revised — 최종 모델

- v3 weighted의 `best.pt`에서 추가 학습했다.
- 오탐 검수 후 수정한 라벨을 사용했다.
- 수정된 학습 표본을 반복 배치해 학습 비중을 높였다.
- 최종 가중치 위치는 `runs/detect/galuxy_finetune_v4_revised/weights/best.pt`이며 가중치 자체는 Git에 포함하지 않는다.

## 평가 재현

```powershell
python scripts/evaluation/evaluate_galuxy.py `
  --images "<평가 이미지 폴더>" `
  --ground-truth "<정답 YOLO 라벨 폴더>" `
  --predictions "<예측 YOLO 라벨 폴더>" `
  --iou 0.5 `
  --output "results/evaluation.json"
```

예측 라벨은 `class_id x_center y_center width height confidence` 형식을 사용한다. 정답 라벨은 confidence가 없는 표준 YOLO 형식을 사용한다.
