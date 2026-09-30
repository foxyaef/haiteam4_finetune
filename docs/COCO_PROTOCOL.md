# COCO 사전학습 모델 1차 실험 프로토콜

## 고정할 사항

모델은 [고정된 공식 RT-DETR v1 R50-vd 소스·COCO 가중치](../upstream-lock.json)입니다. **80클래스 Head까지 원본 가중치를 그대로 엄격하게 적재**하고 학습·파인튜닝·Head 재초기화는 하지 않습니다. 입력은 RGB 640×640 직접 resize 및 `[0,1]` scaling, batch 1입니다. [공통 설정](../configs/coco_phase1.yaml)의 seed는 42, calibration은 COCO val 512장, 평가는 이와 겹치지 않는 COCO val 1,000장입니다. 후처리는 top 300, COCO bbox 평가 maxDets 100을 사용합니다.

이번 4/6/8비트 실험은 `Conv2d`와 `Linear`의 가중치·입력 활성값에 적용하는 **fake quantization 정확도 실험**입니다. 가중치는 출력 채널별 대칭 MinMax, 활성값은 텐서별 비대칭 MinMax입니다. Functional attention MatMul, 정규화, softmax, deformable sampling 등은 FP32로 남습니다. `provenance.json`의 실제 적용 모듈과 미호출 모듈을 확인하고, 이 결과를 전체 연산 INT4 또는 배포 속도로 표현하지 않습니다.

## 다섯 명의 분담

| 담당 | 조건 | 실행 방법 |
|---|---|---|
| 1 | FP32 + 전체 W8A8/W6A6/W4A4 | `coco.cmd run --group baseline`; `--group all --bits 8`, `6`, `4` |
| 2 | Backbone | `--group backbone --bits 8`, `6`, `4` |
| 3 | Hybrid Encoder | `--group encoder --bits 8`, `6`, `4` |
| 4 | Decoder | `--group decoder --bits 8`, `6`, `4` |
| 5 | 분류·박스 Head | `--group head --bits 8`, `6`, `4` |

각 예시에 공통으로 `--device cpu`, `--device xpu` 또는 `--device cuda`를 추가합니다. 비트 수는 **한 번에 하나씩** 실행합니다. 담당자 모두 원본 가중치 SHA256과 `data/coco/manifest.json`의 해시가 같아야 합니다. 평가 코드는 가중치에 직접 재저장하지 않고 실행 때만 fake quantization을 적용하므로 조건별 결과가 서로 독립적입니다. 결과 폴더가 이미 있으면 덮어쓰지 않습니다.

## 지표

| 지표 | 해석 |
|---|---|
| mAP50:95 | COCO bbox AP를 IoU 0.50:0.05:0.95에서 평균. 주 정확도 지표 |
| ΔAP point | `100 × (FP32 mAP50:95 − 실험 mAP50:95)`. 양수면 손실 |
| AP50 | IoU 0.50에서의 탐지 정확도 |
| AP75 | IoU 0.75에서의 더 엄격한 위치 정확도 |
| Class-wise AP | COCO 80개 클래스 중 손실이 집중되는 클래스 확인 |

`metrics.json`의 AP는 0~1, `summary.csv`의 `delta_ap_points`는 0~100 AP point입니다. mAP는 선택한 **평가 1,000장**에만 해당합니다. official val2017 5,000장 전체 결과 또는 독립 test 점수로 표시하지 않습니다.
