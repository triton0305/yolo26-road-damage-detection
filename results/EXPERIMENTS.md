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

AI Hub 기본 학습부터 v4 revised까지의 전체 학습 곡선과 epoch별 원본 지표는 [`results/detection`](detection/README.md)에 정리했다.

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

## Segmentation 도메인 적응

AI Hub 50,133장으로 학습된 `road_segment_yolo26s/weights/best.pt`에서 시작해 직접 촬영 이미지의 수동 폴리곤 라벨만 추가 학습했다. 기존 AI Hub 전체 데이터를 다시 학습하지 않고 사전학습 가중치를 유지한 상태에서 도메인을 보정하는 방식이다.

### 수동 데이터 구성

- 전체 52장
- train 41장 / validation 11장
- 수동 폴리곤 270개
- 차선 268개 / 신축이음부 2개
- 응력완화줄눈 관련 4개 클래스는 수동 표본 없음

클래스 불균형이 매우 크므로 수동 데이터만으로 6개 클래스 전체의 성능을 평가할 수 없다. 특히 신축이음부와 응력완화줄눈 결과는 AI Hub 기반 모델의 영향이 크다.

### 체크포인트 선택

| 모델 | 설정 | 관찰 결과 | 사용 여부 |
|---|---|---|---|
| manual v1 final | 50 epochs, lr0 0.00005, freeze 10 | 차선 반영은 강하지만 수동 데이터 과적합 가능성 | 비교용 |
| manual v2 light | lr0 0.00002, freeze 12, 조기 종료 | 수동 반영이 부족하고 기존 클래스 오탐 증가 | 제외 |
| manual v1 epoch30 | v1의 30 epoch 체크포인트 | AI Hub 특성 유지와 수동 차선 반영의 중간점 | **최종 통합 사용** |

최종 이미지 177장 추론에서 Segmentation confidence 0.50 기준 338개, Detection confidence 0.25 기준 111개가 표시됐다. 이 수치는 정답 라벨과 비교한 평가 지표가 아니라 출력 규모 확인용이다.

세 실험의 전체 학습 곡선과 epoch별 원본 지표는 [`results/segmentation`](segmentation/README.md)에 별도로 정리했다. `manual v1`의 run-level PR 곡선과 혼동행렬은 최종 선택한 `epoch30.pt` 전용 결과가 아니므로 공개 비교 자료에서 제외했다.

## 최종 통합 출력

- 이미지: Segmentation 마스크·박스·태그 위에 Detection 박스·태그 합성
- 영상: 1920×1080, 30fps, H.264 High Profile, 약 8Mbps
- 오디오: 원본 AAC 48kHz 스테레오 보존
- 표시 이름 변경: `절삭보수부파손` → `절삭보수부`, `긴급보수부파손` → `긴급보수부`
- 모델 가중치와 생성 이미지·영상은 Git에 포함하지 않음
