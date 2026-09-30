# COCO RT-DETR 실제 INT8·INT4 PTQ 실험

공식 COCO 사전학습 **RT-DETR v1 R50-vd**를 그대로 사용합니다. COCO val2017의 고정된 calibration 512장과 평가 1,000장으로 FP32, 실제 INT8, 실제 INT4 ONNX 모델의 정확도·파일 크기·코랩 CPU 추론시간·메모리를 비교합니다. BDD100K 재학습은 이 실험에 포함되지 않습니다.

## 가장 간단한 실행

1. [Colab 노트북 열기](https://colab.research.google.com/github/foxyaef/haiteam4_finetune/blob/main/colab/RTDETR_COCO_Eval.ipynb)
2. 셀을 순서대로 실행합니다. 저장소·공식 가중치·COCO 데이터가 내려받아지고, `rtdetr_fp32.onnx`, `rtdetr_int8.onnx`, `rtdetr_int4.onnx`가 만들어집니다.
3. 같은 코랩 CPU 환경에서 세 모델을 평가하고 결과 ZIP을 다운로드합니다.
4. 로컬에서 `python scripts/analyze_real_ptq.py <다운로드한 ZIP>`을 실행해 `summary.csv`, `analysis.md`를 확인합니다.

자세한 명령과 결과 해석은 [실행 안내](docs/COLAB_WORKFLOW.md)에 있습니다. ONNX 파일에는 모델·데이터 식별 정보가 들어 있어, 나중에는 **양자화된 `.onnx` 파일 하나만** 코랩에 올려 다시 평가할 수 있습니다.

## 무엇을 양자화하나

| 모델 | 방법 | 실제 저비트 범위 | 남는 FP32 범위 |
|---|---|---|---|
| INT8 | COCO calibration을 이용한 정적 QDQ PTQ | 지원되는 Conv·MatMul의 가중치와 활성값 | 지원되지 않거나 양자화되지 않은 연산 |
| INT4 | 블록별 packed weight-only PTQ | **상수 가중치 MatMul**의 가중치 | Conv, 활성값, 그 밖의 연산 |

INT4는 **전체 RT-DETR W4A4가 아닙니다.** ONNX Runtime이 지원하는 실제 4비트 정수 가중치와 `MatMulNBits` 연산을 사용합니다. 각 결과에는 변환된 연산 수와 ONNX Runtime이 최적화한 정수 연산 수가 기록됩니다. 지원되지 않은 연산을 4비트로 양자화했다고 주장하지 않습니다. INT8과 INT4는 적용 범위가 다르므로, 어느 쪽이 “4비트라서 더 빠르다/정확하다”처럼 직접 해석하지 마세요.

주 지표는 **mAP50:95와 FP32 대비 ΔAP point**입니다. AP50·AP75·클래스별 AP, ONNX 파일 크기, 추론시간 mean/p50/p95, 처리 FPS, 프로세스 최대 RSS도 저장합니다. 모든 시간·메모리는 ONNX Runtime **CPU** 실행 기준이며 GPU 메모리와 속도는 이 경로에서 측정하지 않습니다. 코랩 CPU의 하드웨어 배정이 달라질 수 있으므로 속도는 같은 세션의 FP32와 양자화 모델끼리 비교하세요.

## 프로젝트 파일

- [`colab/RTDETR_COCO_Eval.ipynb`](colab/RTDETR_COCO_Eval.ipynb): 설치·모델 생성·평가·결과 다운로드
- [`scripts/onnx_real_ptq.py`](scripts/onnx_real_ptq.py): FP32 ONNX export, INT8 정적 PTQ, packed INT4 PTQ
- [`scripts/onnx_eval.py`](scripts/onnx_eval.py): 실제 ONNX 모델의 COCO 정확도·시간·메모리 평가
- [`scripts/analyze_real_ptq.py`](scripts/analyze_real_ptq.py): 다운로드한 결과 ZIP의 로컬 비교
- [`prepare_data.py`](prepare_data.py): COCO val2017 다운로드와 512/1,000장 분리

이전 fake quantization·BDD100K 파일은 저장소에서 정리했습니다. 데이터·가중치·생성 모델은 Git에 포함되지 않으며 [COCO 이용 조건](https://cocodataset.org/#termsofuse)을 확인하세요.
