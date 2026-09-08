# Segmentation 학습 결과

AI Hub 기반 모델과 Galaxy 수동 라벨 추가 학습 실험의 학습 곡선 및 epoch별 원본 지표입니다.

| 실험 | 학습 기록 | 원본 지표 | 설명 |
|---|---|---|---|
| AI Hub base | [`aihub_base_results.png`](aihub_base_results.png) | [`aihub_base_metrics.csv`](aihub_base_metrics.csv) | AI Hub 50,133장 기반 학습 |
| manual v1 | [`manual_v1_results.png`](manual_v1_results.png) | [`manual_v1_metrics.csv`](manual_v1_metrics.csv) | Galaxy 수동 폴리곤 52장 추가 학습, 50 epochs |
| manual v2 light | [`manual_v2_light_results.png`](manual_v2_light_results.png) | [`manual_v2_light_metrics.csv`](manual_v2_light_metrics.csv) | 수동 데이터 영향을 줄인 실험, 18 epochs에서 조기 종료 |

## 수동 데이터 구성

- 전체 이미지: 52장
- 학습/검증 분할: 41장 / 11장
- 전체 Polygon: 270개
- 차선: 268개
- 신축이음부: 2개
- 응력완화줄눈 관련 4개 클래스: 직접 라벨 없음

## 체크포인트 선택

| 모델 | 설정 | 관찰 결과 | 사용 여부 |
|---|---|---|---|
| manual v1 final | 50 epochs, lr0 0.00005, freeze 10 | 차선 반영은 강하지만 소량 수동 데이터에 과적합 가능성 | 비교용 |
| manual v2 light | lr0 0.00002, freeze 12, 조기 종료 | 수동 데이터 반영이 부족하고 기존 클래스 오탐 증가 | 제외 |
| manual v1 epoch30 | v1의 30 epoch 체크포인트 | AI Hub 특성 유지와 수동 차선 반영의 중간점 | **최종 통합 사용** |

## 학습 곡선

### AI Hub base

![AI Hub base 학습 곡선](aihub_base_results.png)

### Galaxy manual v1

![Galaxy manual v1 학습 곡선](manual_v1_results.png)

### Galaxy manual v2 light

![Galaxy manual v2 light 학습 곡선](manual_v2_light_results.png)

## 해석 주의사항

- 수동 라벨 270개 중 차선이 268개이고 신축이음부는 2개입니다.
- 수동 검증셋만으로 기존 6개 클래스 전체의 성능을 대표할 수 없습니다.
- `manual v1 epoch30.pt`는 최고 단일 지표가 아니라 AI Hub 특성 유지와 실제 촬영 차선 반영 사이의 절충점으로 선택했습니다.
- CSV는 재현과 검토를 위한 Ultralytics 원본 epoch 지표입니다.
