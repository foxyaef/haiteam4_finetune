# 이 프로젝트의 RT-DETR 내부 구조

이번 실험은 **RT-DETR v1 R50-vd의 COCO 사전학습 가중치를 수정 없이 사용**한다. 기본 입력은 RGB 640×640, 분류 출력은 COCO 80개 클래스다. BDD100K 파인튜닝이나 Head 재초기화는 하지 않는다. R18이나 RT-DETRv2의 세부 설정을 이 모델에 적용하면 다른 실험이 된다. 정확한 버전·가중치 출처는 [`upstream-lock.json`](../upstream-lock.json)에 고정돼 있다.

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

각 Decoder 층은 query 간 self-attention, 세 스케일 특징을 참조하는 deformable cross-attention, FFN 및 정규화로 구성된다. Cross-attention은 모든 영상 위치를 균등하게 보는 대신 기준점 주변의 몇 위치를 샘플링한다. `sampling_offsets`, `attention_weights`, `value_proj`, `output_proj` 같은 Linear 층이 있으나, 좌표 계산·softmax·샘플링 자체는 별도 연산이다. [고정 버전 Decoder 구현](https://github.com/lyuwenyu/RT-DETR/blob/29320b6fd828f8e0987a71426cf2d961b09dfed7/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py)

## 4 분류와 박스 Head

이 구현에서 Head는 독립적인 최상위 `head` 객체가 아니다. `decoder.enc_score_head`, `decoder.enc_bbox_head`, `decoder.dec_score_head`, `decoder.dec_bbox_head`에 분류·박스 예측 층이 들어 있다. 학습에서는 Decoder 중간 층의 보조 예측과 denoising query도 사용한다. 추론에서는 마지막 출력이 핵심이며 학습 전용 경로는 실행되지 않을 수 있다.

실제 양자화 대상은 PyTorch 모듈 이름이 아니라 **ONNX export 후 그래프의 연산자와 상수 가중치 여부**로 결정된다. 따라서 Head라고 해서 모든 층이 같은 정밀도로 바뀌지는 않는다.

## 5 학습과 추론의 차이

학습은 Hungarian matching으로 예측과 정답을 연결하고 분류·박스·GIoU 손실을 사용한다. 보조 출력과 denoising은 학습 수렴을 돕는다. 추론에서는 학습용 matching과 denoising을 수행하지 않는다. RT-DETR은 query 집합에서 최종 탐지를 바로 내므로 일반적인 YOLO식 NMS를 기본 후처리로 사용하지 않는다. 평가에서는 후처리 top 300 예측을 COCO 방식의 `maxDets=100`으로 집계한다. [공식 모델 설명](https://github.com/lyuwenyu/RT-DETR), [프로젝트 평가 구현](../scripts/metrics.py)

## 6 이번 실제 PTQ의 범위

FP32 모델을 ONNX로 export한 뒤, INT8은 평가 이미지와 겹치지 않는 COCO val 512장으로 calibration한 **Conv·MatMul의 정적 QDQ**를 적용한다. ONNX Runtime CPU 최적화 그래프에 정수 연산이 남는지 확인한다.

INT4는 ONNX Runtime의 **상수 가중치 MatMul**에 한해 블록별 4비트 가중치를 패킹한 `MatMulNBits`를 적용한다. Conv와 활성값은 FP32에 남는다. 따라서 모델 전체 W4A4라고 부르지 않는다. 실제 변환된 연산 수와 파일 크기·메모리·지연시간을 함께 보고한다. [ONNX Runtime 양자화 지원 범위](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#quantize-to-int4uint4)

특히 출력 점수·박스 Head에서 AP 손실이 보이면 분류 순위, 박스 위치, 클래스별 AP가 어떻게 달라졌는지 살펴본다. 한두 이미지의 탐지 결과만으로 원인을 단정하지 않는다.
