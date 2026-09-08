# YOLO26 도로 노면 Detection 및 Segmentation

AI Hub의 고해상도 도로 노면 데이터와 직접 촬영한 Galaxy 이미지를 이용해 포트홀과 균열 등의 도로 손상을 Detection하고, 차선과 신축이음부 등의 도로 노면 요소를 Segmentation하는 프로젝트입니다.

Ultralytics YOLO26s 기반으로 Detection과 Segmentation 모델을 각각 학습하고 실제 촬영 데이터로 추가 학습한 뒤, 최종적으로 두 모델을 동일 이미지와 영상에 적용해 Bounding Box와 Polygon Mask를 함께 출력했습니다.

## 최종 모델

### Detection

* 모델: YOLO26s
* AI Hub 학습 데이터: 159,509장
* AI Hub 검증 데이터: 19,939장
* 직접 Bounding Box 라벨링: 184장 / 163개
* 최종 체크포인트: `runs/detect/galuxy_finetune_v4_revised/weights/best.pt`
* Confidence threshold: `0.25`

### Segmentation

* 모델: YOLO26s-seg
* AI Hub 학습 데이터: 50,133장
* AI Hub 검증 데이터: 6,265장
* 직접 Polygon 라벨링: 52장 / 270개
* 최종 체크포인트: `galuxy_segment_manual_v1/weights/epoch30.pt`
* Confidence threshold: `0.50`

모델 가중치와 전체 학습 출력은 용량 문제로 Git에 포함하지 않습니다. 실험별 변경점과 평가 기준은 [`results/EXPERIMENTS.md`](results/EXPERIMENTS.md), 선별한 학습 곡선과 epoch별 원본 지표는 [`results/detection`](results/detection/README.md)과 [`results/segmentation`](results/segmentation/README.md)에 정리되어 있습니다.

## 프로젝트 파이프라인

```text
                         AI Hub 도로 노면 데이터
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                Detection                  Segmentation
                    │                           │
          YOLO Detection 변환          YOLO Polygon 변환
                    │                           │
             기본 모델 학습               기본 모델 학습
                    │                           │
          Galaxy Bounding Box          Galaxy Polygon 직접 라벨링
                    │                           │
           v3 weighted 학습              manual v1 / v2 학습
                    │                           │
        오탐 검수 및 라벨 수정           체크포인트 비교
                    │                           │
           v4 revised 학습                 epoch30 선택
                    │                           │
             정량 성능 평가              실환경 추론 검수
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                         Detection + Segmentation
                              통합 추론
                                  │
                      이미지 / 영상 결과 출력
```

최종 단계에서는 동일한 입력 이미지와 영상에 두 모델을 각각 적용한 뒤, Detection의 Bounding Box와 Segmentation의 Polygon Mask를 하나의 결과로 합성했습니다.

## 🎥 Demo Video

[![Road Surface Detection Demo](https://img.youtube.com/vi/BOlS7ECXjTI/0.jpg)](https://www.youtube.com/watch?v=BOlS7ECXjTI)

> AI Hub 고해상도 도로노면 이미지 데이터로 학습한 모델의 실도로 환경 추론 결과입니다.

### Detection
<img width="5712" height="3212" alt="det_picture_093" src="https://github.com/user-attachments/assets/fa2a9775-c140-4ecb-9ada-729e01a59654" />
<img width="5712" height="3212" alt="det_picture_152" src="https://github.com/user-attachments/assets/fb90eb62-eb99-40b5-b76f-00abdba79b65" />



### Segmentation
<img width="5712" height="3212" alt="seg_picture_093" src="https://github.com/user-attachments/assets/aea8f81a-55e4-4532-abcb-d9aaa2144c36" />
<img width="5712" height="3212" alt="seg_picture_152" src="https://github.com/user-attachments/assets/1e55b76c-9042-44ac-b203-21c4fc766edd" />

### Detection + Segmentation
<img width="5712" height="3212" alt="picture_093" src="https://github.com/user-attachments/assets/ee6f54dd-c0c9-4d70-96ce-7be50527ce80" />
<img width="5712" height="3212" alt="picture_152" src="https://github.com/user-attachments/assets/9606959c-b63d-4ef2-a0bf-61a8f80a0bdf" />

## Detection

AI Hub 도로 노면 데이터를 YOLO Detection 형식으로 변환해 기본 모델을 학습하고, 직접 촬영한 Galaxy 데이터를 추가해 실제 촬영 환경에 맞게 추가 학습했습니다.

Galaxy 데이터의 비중을 높인 v3 모델을 학습한 뒤 실제 이미지의 오탐을 검수하고 라벨을 수정해 최종 v4 모델을 학습했습니다.

| Metric       | Result |
| ------------ | -----: |
| Test Images  |    184 |
| Ground Truth |    163 |
| TP           |     92 |
| FP           |     24 |
| FN           |     71 |
| Precision    |  0.793 |
| Recall       |  0.564 |
| F1-score     |  0.659 |

평가는 예측 클래스와 정답 클래스가 동일하고 IoU가 0.5 이상인 경우 TP로 판정했습니다.

## Segmentation

AI Hub Segmentation 데이터 50,133장으로 학습한 YOLO26s-seg 모델을 기반으로, 직접 촬영한 Galaxy 이미지 52장을 Polygon 방식으로 라벨링해 추가 학습했습니다.

* 전체 Polygon: 270개
* 차선: 268개
* 신축이음부: 2개
* Segmentation 클래스: 6개
* 실험: `galuxy_segment_manual_v1`, `galuxy_segment_manual_v2_light`

v1은 학습 후반부에서 소량의 Galaxy 데이터에 과적합되는 경향이 있었고, v2는 AI Hub 기본 모델의 특성이 지나치게 유지되면서 실제 촬영 이미지에서 신축이음부 오탐이 많이 발생했습니다.

따라서 두 특성의 절충점으로 `galuxy_segment_manual_v1/weights/epoch30.pt`를 최종 체크포인트로 선택했습니다.

직접 라벨링한 270개 Polygon 중 268개가 차선이고 신축이음부는 2개이며, 나머지 응력완화줄눈 계열에는 직접 라벨이 없습니다. 따라서 이번 Segmentation 실험은 6개 클래스 전체의 성능 향상보다는 **AI Hub 기반 모델을 실제 Galaxy 촬영 환경에 적응시킨 소규모 Domain Adaptation 실험**으로 해석했습니다.

별도의 Segmentation Ground Truth 기반 정량 평가는 수행하지 않았습니다.

## Detection + Segmentation 통합 추론

최종 단계에서는 동일한 입력에 Detection 모델과 Segmentation 모델을 각각 실행하고 두 결과를 하나의 이미지와 영상으로 출력했습니다.

```text
Input Image
    │
    ├── YOLO26s Detection
    │      └── Bounding Box
    │
    └── YOLO26s-seg
           └── Polygon Mask
                  │
                  ▼
              Result Merge
                  │
                  ▼
          Final Image / Video
```

최종 이미지 추론 결과:

* 입력 이미지: 177장
* Detection 검출: 111개
* Segmentation 검출: 338개

  * 차선: 330개
  * 신축이음부: 4개
  * 응력완화줄눈 계열: 4개

Segmentation의 338개는 정답과 비교한 성능 지표가 아니라 모델이 최종 이미지에서 출력한 전체 Prediction 수입니다.

최종 결과에는 Detection Bounding Box와 Segmentation Polygon Mask, 클래스 이름과 Confidence를 함께 표시했습니다.

영상에도 동일한 방식을 적용해 실제 재생시간을 유지하는 1080p 30fps H.264 MP4로 출력했습니다.

## 프로젝트 구조

```text
configs/              데이터셋 YAML 설정
scripts/data/         데이터 변환과 학습 데이터 준비
scripts/training/     모델 학습 및 추가 학습
scripts/inference/    모델 추론
scripts/evaluation/   TP·FP·FN 기반 Detection 평가
results/              실험 기록, 선별 학습 곡선과 epoch별 지표
models/pretrained/    로컬 사전학습 가중치, Git 제외
runs/                 학습 체크포인트와 그래프, Git 제외
```

## 주요 실행 파일

| 파일                                                | 역할                                |
| ------------------------------------------------- | --------------------------------- |
| `scripts/data/convert_road_labels.py`             | AI Hub 라벨을 YOLO 형식으로 변환           |
| `scripts/data/prepare_galuxy_weighted_dataset.py` | Galaxy 비중을 높인 v3 데이터셋 준비          |
| `scripts/training/train_galuxy_weighted.py`       | v3 weighted 모델 학습                 |
| `scripts/training/train_galuxy_revised.py`        | 수정 라벨로 최종 v4 Detection 모델 학습      |
| `scripts/training/train_galuxy_segment_manual.py` | 수동 Polygon Segmentation 추가 학습          |
| `scripts/inference/predict_galuxy_weighted.py`    | 이미지 일괄 추론과 예측 라벨 저장               |
| `scripts/inference/predict_combined_final.py`     | 이미지 Detection + Segmentation 통합 추론    |
| `scripts/inference/predict_combined_video.py`     | 영상 통합 추론 및 1080p30 출력                |
| `scripts/evaluation/evaluate_galuxy.py`           | 같은 클래스·IoU 0.5 기준 Detection 성능 평가 |

## 개발 환경

* Python 3.11.15
* Ultralytics YOLO11 / YOLO26s / YOLO26s-seg
* PyTorch
* OpenCV
* Pillow
* PyYAML

## 데이터 출처

* 데이터셋: AI Hub 고해상도 도로 노면 데이터
* 수행기관: 에이치씨아이플러스(주)
* 제공기관: 과학기술정보통신부 / 한국지능정보사회진흥원(NIA)
* 링크: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71781
