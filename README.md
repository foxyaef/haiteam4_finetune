# RT-DETR COCO 공통 실험 환경

팀원은 **로컬 PC에서 양자화**하고, 완성한 ONNX 모델 **한 개**를 **Colab에서 평가**합니다. 평가 조건은 공식 COCO 사전학습 RT-DETR v1 R50-vd, 입력 640×640, COCO val2017 고정 부분집합 1,000장, ONNX Runtime CPU입니다. 양자화 방법 자체는 팀원이 선택합니다.

## 1. 로컬: 양자화 모델 만들기

Windows PowerShell 예시입니다. Python 3.11을 설치한 뒤 실행하세요. GPU를 사용할 경우 아래 PyTorch 설치 줄 대신 [공식 설치 페이지](https://pytorch.org/get-started/locally/)에서 자신의 장치에 맞는 명령을 골라 **가상환경 안에서** 실행하세요.

```powershell
git clone https://github.com/foxyaef/haiteam4_finetune.git
cd haiteam4_finetune
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch torchvision
python -m pip install -r requirements.txt -r requirements-onnx.txt
python scripts/colab_setup.py
python prepare_data.py
python scripts/onnx_real_ptq.py export --output artifacts/rtdetr_fp32.onnx
```

`prepare_data.py`는 겹치지 않는 **calibration 512장**과 **평가 1,000장**을 고정합니다. 로컬 PTQ는 calibration 512장을 사용하세요. 예제 INT8/INT4 변환은 [로컬 양자화 안내](docs/LOCAL_QUANTIZATION.md)에 있습니다. 각자 만든 모델은 입력·출력 형식이 [공통 인터페이스](docs/COLAB_WORKFLOW.md#모델-제출-조건)와 같아야 합니다.

## 2. Colab: 동일 조건에서 평가

[공통 평가 노트북 열기](https://colab.research.google.com/github/foxyaef/haiteam4_finetune/blob/main/colab/RTDETR_COCO_Eval.ipynb) → 셀을 위에서 아래로 실행 → 로컬 ONNX 파일 하나 업로드 → 결과 ZIP 다운로드. 노트북은 **양자화를 실행하지 않습니다.** 같은 세션에서 FP32 기준 모델과 업로드 모델을 평가합니다.

결과에는 mAP50:95, AP50, AP75, 클래스별 AP, FP32 대비 ΔAP, ONNX 파일 크기, 추론시간 mean/p50/p95, 관측 FPS, 최대 프로세스 RSS, 런타임 연산 수가 포함됩니다. 시간과 메모리는 **Colab CPU** 기준이며 세션이 다르면 CPU가 달라질 수 있습니다.

## 3. 로컬: 결과 비교

```powershell
python scripts/analyze_real_ptq.py .\coco_eval_results_YYYYMMDD_HHMMSS.zip
```

`runs/real_ptq_analysis/summary.csv`와 `analysis.md`가 생성됩니다. ΔAP point = (FP32 mAP50:95 − 제출 모델 mAP50:95) × 100입니다.

**제출 모델의 이름이 INT4라도 전체 RT-DETR이 4비트라는 뜻은 아닙니다.** 예제 INT4는 상수 가중치 MatMul만 실제 4비트이고 Conv·활성값은 FP32입니다. 자체 제작 모델은 출처와 방법을 팀이 선언한 것으로 표시합니다. 그래프 연산 수와 생성 기록을 함께 확인해야 실제 양자화 범위를 주장할 수 있습니다.

자세한 실행·업로드·결과 해석은 [Colab 평가 안내](docs/COLAB_WORKFLOW.md), [데이터 구성](docs/COCO_DATA.md)을 참고하세요. 데이터·가중치·모델 파일은 GitHub에 포함하지 않습니다.
