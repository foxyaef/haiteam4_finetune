# 이 프로젝트의 RT-DETR 내부 구조

이 저장소는 **RT-DETR v1 R50-vd**를 사용한다. 기본 입력은 RGB 640×640이며 COCO 사전학습 가중치를 BDD100K 10개 탐지 클래스로 미세조정한다. R18이나 RT-DETRv2의 세부 설정을 이 모델에 적용하면 다른 실험이 된다. 정확한 버전·가중치 출처는 [`upstream-lock.json`](../upstream-lock.json)에 고정돼 있다.

```text
RGB 이미지 [B,3,640,640]
        ↓
PResNet-50-vd Backbone
        ├─ S3: stride 8,  약 80×80,  512채널
        ├─ S4: stride 16, 약 40×40, 1024채널
        └─ S5: stride 32, 약 20×20, 2048채널
        ↓
Hybrid Encoder: 채널을 256으로 맞춤
        ├─ AIFI: S5에만 intra-scale self-attention
        └─ CCFF: 세 스케일 특징을 위→아래·아래→위로 융합
        ↓
Encoder 출력 특징 → query 후보 점수·박스 추정 → 300개 후보 선택
        ↓
Transformer Decoder: 6개 층
        ├─ query 간 self-attention
        ├─ 다중 스케일 deformable cross-attention
        └─ FFN + 정규화 + 반복적인 박스 갱신
        ↓
분류 Head와 박스 Head → 최종 300개 query 예측
        ↓
좌표 복원·점수 변환·top query 선택 → 2D 객체 탐지 결과
```

크기는 640×640 입력과 stride 설정에서 나온 **설계상 특징맵 크기**이며, 구체적 tensor shape는 실행 시 확인한다. 위 수치는 [프로젝트가 고정한 공식 R50 설정](https://github.com/lyuwenyu/RT-DETR/blob/29320b6fd828f8e0987a71426cf2d961b09dfed7/rtdetr_pytorch/configs/rtdetr/include/rtdetr_r50vd.yml)에 따른다.

## 1 Backbone

PResNet은 영상에서 위치와 의미 정보를 추출한다. 앞단의 공간 해상도가 높은 S3는 작은 물체에, 깊은 S5는 넓은 문맥에 유용하다. R50-vd는 ResNet-50의 변형 stem과 downsampling 방식을 사용한다. 이 프로젝트는 `return_idx: [1,2,3]`으로 세 수준의 특징을 꺼낸다. Backbone을 양자화하면 이후 모든 부분에 공급되는 특징이 달라지므로 오류가 뒤로 전파될 수 있다.

## 2 Hybrid Encoder

일반 Transformer가 모든 스케일에서 self-attention을 실행하면 비용이 커진다. RT-DETR의 Efficient Hybrid Encoder는 **AIFI**로 가장 깊은 S5 특징의 같은 스케일 내부 관계를 처리하고, **CCFF**로 스케일 간 정보를 결합한다. AIFI에는 attention과 FFN이 있고, 융합 경로에는 convolution·업샘플·결합 연산이 있다. AIFI의 query/key/value 계산과 CCFF의 Conv는 오차 분포와 연산량이 달라, Encoder가 민감하면 두 하위 부분을 나누어 분석할 수 있다. [RT-DETR 논문](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.pdf)

## 3 Query 선택과 Decoder

Encoder의 특징마다 분류 점수와 박스 후보를 만들고, 상위 후보를 초기 query로 고른다. query의 초기 품질은 이후 Decoder가 물체를 찾는 난이도에 영향을 준다. 공식 R50 설정에서는 **300개 query**와 **6개 Decoder 층**을 사용한다.

각 Decoder 층은 query 간 self-attention, 세 스케일 특징을 참조하는 deformable cross-attention, FFN 및 정규화로 구성된다. Cross-attention은 모든 영상 위치를 균등하게 보는 대신 기준점 주변의 몇 위치를 샘플링한다. `sampling_offsets`, `attention_weights`, `value_proj`, `output_proj` 같은 Linear 층이 있으나, 좌표 계산·softmax·샘플링 자체는 별도 연산이다. 따라서 Linear만 양자화하는 1차 실험에서는 Decoder 전체 연산이 저비트가 되는 것은 아니다. [고정 버전 Decoder 구현](https://github.com/lyuwenyu/RT-DETR/blob/29320b6fd828f8e0987a71426cf2d961b09dfed7/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py)

## 4 분류와 박스 Head

이 구현에서 Head는 독립적인 최상위 `head` 객체가 아니다. `decoder.enc_score_head`, `decoder.enc_bbox_head`, `decoder.dec_score_head`, `decoder.dec_bbox_head`에 분류·박스 예측 층이 들어 있다. 학습에서는 Decoder 중간 층의 보조 예측과 denoising query도 사용한다. 추론에서는 마지막 출력이 핵심이며 학습 전용 경로는 실행되지 않을 수 있다.

팀의 구성요소 민감도 표에서는 예측용 네 접두사를 **Head**에 배정하고, Decoder의 나머지를 **Decoder**에 배정한다. 이 규칙은 [`scripts/official.py`](../scripts/official.py)의 `component()`와 같다. 경계에 있는 query 선택 부분의 분류·박스 층은 Head로, query 생성·갱신의 다른 대상 층은 Decoder로 구분한다. 동일 층을 두 그룹에 중복 배정하지 않는다.

## 5 학습과 추론의 차이

학습은 Hungarian matching으로 예측과 정답을 연결하고 분류·박스·GIoU 손실을 사용한다. 보조 출력과 denoising은 학습 수렴을 돕는다. 추론에서는 학습용 matching과 denoising을 수행하지 않는다. RT-DETR은 query 집합에서 최종 탐지를 바로 내므로 일반적인 YOLO식 NMS를 기본 후처리로 사용하지 않는다. 평가에서는 후처리 top 300 예측을 COCO 방식의 `maxDets=100`으로 집계한다. [공식 모델 설명](https://github.com/lyuwenyu/RT-DETR), [프로젝트 평가 구현](../scripts/metrics.py)

## 6 이번 양자화 실험에서 실제로 바뀌는 부분

1차 Basic PTQ는 실행 중 호출되는 `torch.nn.Conv2d`와 `torch.nn.Linear`에 한정한다. 가중치에는 **출력 채널별 대칭 MinMax**, 입력 활성값에는 **텐서별 비대칭 MinMax**를 적용한다. 512장 train calibration의 FP32 입력 분포를 관측한 뒤 scale을 고정한다. 각 담당자는 자기 영역만 W8A8·W6A6·W4A4로 바꾸고 나머지는 FP32로 둔다.

PyTorch `MultiheadAttention` 안의 `in_proj_weight`처럼 `Linear.forward` 호출을 거치지 않는 가중치와 functional attention MatMul, BatchNorm·LayerNorm, softmax, deformable sampling, bias·누산기는 이 단계의 양자화 대상이 아니다. 실행 결과의 `provenance.json`에 실제 호출·양자화된 모듈 수와 실행되지 않은 모듈 목록을 남긴다. **W4A4는 지정한 Conv/Linear 입력·가중치의 표현 정밀도**를 뜻한다. 현재 코드는 fake quantization으로 정확도를 검사하며 실제 INT4 저장·연산·라즈베리파이 속도를 평가하지 않는다.

특히 출력 점수·박스 Head에서 큰 AP 손실이 보이면 분류 순위, 박스 위치, 클래스별 AP가 어떻게 달라졌는지 살펴본다. Backbone 또는 Encoder가 민감하면 해당 특징을 받는 모든 Head 예측이 함께 흔들릴 수 있으므로 한두 이미지의 탐지 결과로 원인을 단정하지 않는다.
