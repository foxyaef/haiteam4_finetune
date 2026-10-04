# RT-DETR 공통 모델 평가 환경

팀원이 로컬에서 만든 RT-DETR 모델을 **Colab에 업로드하고 실제 COCO 이미지를 추론**하여 비교합니다. 이 저장소는 양자화 코드를 제공하지 않습니다. 모델을 만들 때는 같은 COCO 사전학습 RT-DETR v1 R50-vd 체크포인트, calibration 이미지와 평가 이미지를 사용합니다.

## 가장 빠른 실행 순서

1. [Colab 평가 노트북 열기](https://colab.research.google.com/github/foxyaef/haiteam4_finetune/blob/main/colab/COCO_Eval.ipynb). 다운로드한 `COCO_Eval.ipynb`를 Colab에 직접 업로드해도 됩니다. GPU 런타임이면 `provider=cuda`, CPU면 `cpu`를 선택합니다.
2. **평가 설정 → 환경과 COCO 이미지 준비** 셀을 실행합니다. 기본 `full`은 공통 평가 이미지 1,000장입니다. 먼저 `smoke`를 선택하면 같은 목록의 3장으로 실행을 확인합니다. 최초 다운로드는 약 1.1GB입니다. 평가 전용 환경을 사용하므로 Colab의 기존 ONNX Runtime·NumPy와 섞이지 않습니다.
3. **모델 파일 선택** 셀에서 `upload`로 PC 파일을 올리거나 `drive`로 Google Drive 폴더를 지정합니다. Hybrid Encoder 실험은 `artifacts/cal512/`의 `fp32.onnx`와 아래 6개 파일을 함께 선택하세요. 각 모델은 가중치가 포함된 단일 ONNX 파일이어야 합니다.
4. **평가·결과 다운로드** 셀을 실행합니다. 표·탐지 미리보기와 ZIP이 만들어집니다. 모델 하나가 실패해도 나머지를 계속 평가하고 실패 로그도 ZIP에 담습니다. 성공한 `fp32.onnx`가 있으면 ΔmAP와 크기 비율을 계산합니다.

| 양자화 범위 | W8A8 | W4A16 |
| --- | --- | --- |
| Transformer만 | `encoder_transformer_w8a8.onnx` | `encoder_transformer_w4a16.onnx` |
| CNN만 | `encoder_cnn_w8a8.onnx` | `encoder_cnn_w4a16.onnx` |
| 둘 다 | `encoder_both_w8a8.onnx` | `encoder_both_w4a16.onnx` |

다른 component 모델도 같은 입출력 계약이면 평가할 수 있습니다. 노트북에는 평가 코드 스냅샷과 해시가 포함되어 있어, 실행할 때 GitHub 최신 파일이나 기존 clone 상태에 영향을 받지 않습니다. 수정한 평가 코드를 배포할 때는 `python scripts/build_colab.py`로 노트북을 다시 생성하세요.

`.pth` 체크포인트는 그대로 업로드할 수 없습니다. 먼저 [공식 RT-DETR의 `tools/export_onnx.py`](https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/tools/export_onnx.py) 방식으로 deploy ONNX를 만드세요. 양자화 모델도 같은 **입출력 계약**을 유지해야 합니다: `images` `[1,3,640,640]` RGB 0–1 float32/float16, `orig_target_sizes` `[1,2]` 원본 `[width,height]` int64/int32; 출력 `labels` (연속 COCO 라벨 0–79), `boxes` (원본 이미지 픽셀의 xyxy), `scores` (0–1). 다른 엔진이나 원시 DETR 출력은 변환 후 사용할 수 있습니다. 모델별 ONNX 파일을 같은 Colab 런타임·같은 실행 provider에서 비교하세요.

## 무엇을 측정하나

| 항목 | 의미 |
| --- | --- |
| mAP50:95, AP50, AP75, 클래스별 AP | 고정 COCO val2017 **1,000장 부분집합**에서 pycocotools bbox 평가. 공식 5,000장 수치와 다릅니다. |
| 실제 파일 크기 | 업로드한 `.onnx` 파일의 바이트 수. `.pth` 크기나 이론적 비트 수가 아닙니다. |
| 최대 RSS / RSS 증가분 | 모델 로드·워밍업·추론 중 해당 평가 프로세스의 RAM 사용량을 20ms 간격으로 샘플링한 값. 런타임·입출력·예측 데이터도 포함합니다. 그래프 검사와 별도 profiling은 제외합니다. |
| GPU 프로세스 메모리 | CUDA에서 NVML이 허용하면 해당 프로세스의 GPU 할당량을 샘플링합니다. 불가하면 `null`과 이유를 기록합니다. |
| 추론 p50/p95, FPS | batch 1, 별도 10장 워밍업 후 1,000장 모두의 `session.run` 시간. CUDA 전송 시간도 포함합니다. 전처리·후처리를 포함한 별도 end-to-end 시간도 기록합니다. |
| 양자화 그래프 검사 | ONNX의 Q/DQ·정수 연산·`MatMulNBits`·INT4 텐서 수를 기록합니다. **파일명이나 검사 결과만으로 W8A8/W4A16 전체 적용을 증명하지는 못합니다.** |
| 실제 연산 배치 | 별도 1장 profiling에서 CPU/CUDA kernel event와 연산 종류를 기록합니다. 지연·메모리 측정에는 profiling 비용을 포함하지 않습니다. |

`latency_repeats=3`이면 이미지마다 시간 측정을 3회 반복합니다. AP는 첫 번째 예측만 사용하며 중복 탐지를 넣지 않습니다. **3장 smoke의 AP·속도는 최종 연구 결과로 사용하지 마세요.** AP는 0~1, `delta_mAP_pp`는 percentage point, 크기는 MiB입니다. CUDA의 FP32 기준 비교를 위해 TF32를 끄고 CPU thread 수를 공통 설정으로 고정합니다.

현재 만든 W4A16은 실제 INT4 텐서를 FP16으로 복원해 Conv/MatMul을 실행합니다. INT4 저장과 전용 INT4 가속 커널 사용은 각각 확인해야 합니다. CUDA에서도 일부 노드는 CPU로 배치될 수 있으므로 `provider_placement.json`과 표의 `cuda_uses_cpu_nodes`를 확인하세요. 지원되지 않는 모델은 실패로 기록하며 다른 정밀도의 결과로 대체하지 않습니다. GPU 메모리를 측정하지 못하면 `null`과 이유를 남깁니다. Colab GPU 종류·부하가 바뀌면 성능 수치도 달라집니다.

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

여러 모델을 일괄 평가하려면 `scripts/evaluate_suite.py --models 모델1.onnx 모델2.onnx --output results/새실험폴더 --provider cpu`를 사용합니다. `--max-images 3`은 실행 확인, `--repeats 3`은 지연 반복 측정입니다. 이미 준비된 외부 COCO 폴더는 `--data-dir C:\path\to\data\coco`로 지정할 수 있습니다.

CUDA는 `requirements-eval-gpu.txt`를 설치하고 `--provider cuda`를 지정합니다. `onnxruntime` CPU 패키지와 `onnxruntime-gpu`는 **같은 Python 환경에 함께 설치하지 마세요**. `uploaded_models/`, `data/`, `results/`, 가중치와 ONNX 모델은 Git 추적에서 제외됩니다.

```text
protocol.json                    공통 모델·데이터·입력·측정 조건
upstream-lock.json               공식 소스 커밋·체크포인트 SHA256
prepare_data.py                  COCO 다운로드·부분집합 생성·검증
scripts/setup_model.py           공식 모델 다운로드·검증
scripts/evaluate_model.py        업로드한 ONNX로 실제 이미지 추론·측정
scripts/evaluate_suite.py        모델별 별도 프로세스·실패 처리·요약·ZIP
scripts/model_audit.py           ONNX 양자화 연산 증거 기록
scripts/metrics.py               COCO AP 및 클래스별 지표
scripts/build_colab.py           평가 코드 스냅샷을 담은 노트북 생성
colab/COCO_Eval.ipynb            모델 업로드·일괄 비교·결과 다운로드
```

ZIP에는 `summary.csv/json`, `errors.json`, `logs/`, 환경 패키지 목록과 모델별 `metrics.json`, `class_metrics.csv`, `quantization_audit.json`, `provider_placement.json`, `runtime_profile.json`, `provenance.json`, `predictions.json`, `preview_*.jpg`가 들어갑니다. 전체 CSV에는 AP50·AP75와 end-to-end 지표도 포함됩니다. 예측 JSON만 보유한 경우 `scripts/evaluate_predictions.py`로 정확도만 계산할 수 있습니다.

수정본은 실제 FP32와 6개 양자화 모델을 각각 3장씩 CPU에서 평가해 추론·AP 계산·RAM/지연 기록·profiling·ZIP 생성을 확인했습니다. 새 환경에서 노트북 코드셀의 전체 흐름도 확인하며, Colab 브라우저의 업로드/다운로드 UI와 CUDA 실기 검증은 이 로컬 검사 범위에 포함되지 않습니다. [검증 기록](colab/VALIDATION.md)에 범위를 명시합니다.

구현 참고: [ONNX Runtime CUDA 의존성·preload·TF32](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html), [ORT profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html), [COCO 이용 조건](https://cocodataset.org/#termsofuse).
