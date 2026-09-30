# RT-DETR 공통 모델 평가 환경

팀원이 로컬에서 만든 RT-DETR 모델을 **Colab에 업로드하고 실제 COCO 이미지를 추론**하여 비교합니다. 이 저장소는 양자화 코드를 제공하지 않습니다. 모델을 만들 때는 같은 COCO 사전학습 RT-DETR v1 R50-vd 체크포인트, calibration 이미지와 평가 이미지를 사용합니다.

## 가장 빠른 실행 순서

1. [Colab 평가 노트북 열기](https://colab.research.google.com/github/foxyaef/haiteam4_finetune/blob/main/colab/COCO_Eval.ipynb). 가능하면 GPU 런타임을 선택하고 첫 셀의 `provider='cuda'`를 유지합니다. CPU 런타임이면 `cpu`로 바꿉니다.
2. 첫 셀을 실행해 의존성과 COCO val2017 부분집합을 받습니다. 다운로드는 약 1.1GB입니다.
3. `fp32.onnx`와 비교할 양자화 모델 (`backbone_w8a8.onnx`, `backbone_w4a16.onnx` 등)을 업로드합니다. 모두 **단일 self-contained ONNX 파일**이어야 합니다. 파일이 크면 노트북에 안내한 Google Drive 경로를 사용할 수 있습니다.
4. 마지막 셀을 실행하면 각 모델로 같은 평가 이미지 1,000장을 실제 추론합니다. 표와 탐지 미리보기가 나오고, 지표·클래스별 AP·예측·모델 검사 결과를 담은 ZIP이 다운로드됩니다. `fp32.onnx`가 있으면 ΔmAP와 크기 비율도 계산됩니다.

`.pth` 체크포인트는 그대로 업로드할 수 없습니다. 먼저 [공식 RT-DETR의 `tools/export_onnx.py`](https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/tools/export_onnx.py) 방식으로 deploy ONNX를 만드세요. 양자화 모델도 같은 **입출력 계약**을 유지해야 합니다: `images` `[1,3,640,640]` RGB 0–1 float32/float16, `orig_target_sizes` `[1,2]` 원본 `[width,height]` int64/int32; 출력 `labels` (연속 COCO 라벨 0–79), `boxes` (원본 이미지 픽셀의 xyxy), `scores` (0–1). 다른 엔진이나 원시 DETR 출력은 변환 후 사용할 수 있습니다. 모델별 ONNX 파일을 같은 Colab 런타임·같은 실행 provider에서 비교하세요.

## 무엇을 측정하나

| 항목 | 의미 |
| --- | --- |
| mAP50:95, AP50, AP75, 클래스별 AP | 고정 COCO val2017 **1,000장 부분집합**에서 pycocotools bbox 평가. 공식 5,000장 수치와 다릅니다. |
| 실제 파일 크기 | 업로드한 `.onnx` 파일의 바이트 수. `.pth` 크기나 이론적 비트 수가 아닙니다. |
| 최대 RSS / RSS 증가분 | 모델 로드와 추론 중 해당 평가 프로세스의 RAM 사용량을 20ms 간격으로 샘플링한 값. 런타임·입출력·예측 데이터도 포함합니다. |
| GPU 프로세스 메모리 | CUDA에서 NVML이 허용하면 해당 프로세스의 GPU 할당량을 샘플링합니다. 불가하면 `null`과 이유를 기록합니다. |
| 추론 p50/p95, FPS | batch 1, 첫 10장 워밍업 후 `session.run` 시간. 전처리·후처리를 포함한 별도 end-to-end 시간도 기록합니다. |
| 양자화 그래프 검사 | ONNX의 Q/DQ·정수 연산·`MatMulNBits`·INT4 텐서 수를 기록합니다. **파일명이나 검사 결과만으로 W8A8/W4A16 전체 적용을 증명하지는 못합니다.** |

W4A16 지원은 양자화된 모델의 연산과 ONNX Runtime 실행 provider에 달려 있습니다. 특히 INT4 weight-only `MatMulNBits`가 있다고 해서 모든 Conv/Backbone 가중치가 4비트인 것은 아닙니다. CUDA를 선택해도 일부 노드는 CPU에서 실행될 수 있으므로 `provider=cuda`만으로 전체 GPU 실행을 증명하지 않습니다. 지원되지 않는 모델은 평가가 실패하며, 결과를 FP32나 W8A8로 대체해 표시하지 않습니다. Colab GPU의 종류·부하가 바뀌면 속도와 메모리 수치도 바뀝니다.

## 공통 데이터와 로컬 실행

```powershell
git clone https://github.com/foxyaef/haiteam4_finetune.git
cd haiteam4_finetune
python scripts/setup_model.py
python prepare_data.py
python prepare_data.py --verify-only
```

`setup_model.py`는 [고정된 공식 체크포인트와 소스](upstream-lock.json)를 받고 SHA256을 검증합니다. `prepare_data.py`는 COCO val2017에서 겹치지 않는 calibration 512장과 평가 1,000장을 추출하고 해시를 기록합니다. [protocol.json](protocol.json)에 이미지 크기·전처리·측정 조건이 고정됩니다. 학습·양자화 때는 **calibration 512장만** 사용하고 평가 1,000장은 방법 선택에 쓰지 마세요.

Colab 없이 동일한 모델 평가기를 로컬에서 실행할 수도 있습니다.

```powershell
python -m pip install -r requirements-eval-cpu.txt
python scripts/evaluate_model.py --model C:\path\to\model.onnx --output results\model_cpu --provider cpu
```

CUDA는 `requirements-eval-gpu.txt`를 설치하고 `--provider cuda`를 지정합니다. `onnxruntime` CPU 패키지와 `onnxruntime-gpu`는 **같은 Python 환경에 함께 설치하지 마세요**. `uploaded_models/`, `data/`, `results/`, 가중치와 ONNX 모델은 Git 추적에서 제외됩니다.

```text
protocol.json                    공통 모델·데이터·입력·측정 조건
upstream-lock.json               공식 소스 커밋·체크포인트 SHA256
prepare_data.py                  COCO 다운로드·부분집합 생성·검증
scripts/setup_model.py           공식 모델 다운로드·검증
scripts/evaluate_model.py        업로드한 ONNX로 실제 이미지 추론·측정
scripts/model_audit.py           ONNX 양자화 연산 증거 기록
scripts/metrics.py               COCO AP 및 클래스별 지표
colab/COCO_Eval.ipynb            모델 업로드·일괄 비교·결과 다운로드
```

결과 폴더에는 `metrics.json`, `class_metrics.csv`, `quantization_audit.json`, `provenance.json`, `predictions.json`, `preview_*.jpg`가 생성됩니다. 예측 JSON만 이미 보유한 경우에는 `scripts/evaluate_predictions.py`로 정확도만 별도 계산할 수 있지만, 그 경로로는 파일 크기·메모리·지연을 측정하지 않습니다. [COCO 이용 조건](https://cocodataset.org/#termsofuse)을 확인하세요.
