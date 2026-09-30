# 실제 PTQ 실험: Colab 실행·결과 분석

이 문서는 **실제 ONNX 양자화 모델**의 실험 경로입니다. 이전 fake quantization `.pth`와 결과를 혼합하지 마세요. 공통 기준은 공식 COCO 사전학습 RT-DETR v1 R50-vd, 입력 640×640, COCO val2017 calibration 512장과 평가 1,000장입니다. COCO 공식 val 5,000장 전체 점수와 직접 비교하면 안 됩니다.

## 1. Colab에서 한 번에 실행

[노트북 열기](https://colab.research.google.com/github/foxyaef/haiteam4_finetune/blob/main/colab/RTDETR_COCO_Eval.ipynb) 후 위에서 아래로 실행합니다. 노트북은 저장소를 복제하고 `requirements.txt`와 `requirements-onnx.txt`를 설치하며, `colab_setup.py`로 고정된 공식 RT-DETR 소스와 사전학습 가중치를 준비합니다. `prepare_data.py`가 COCO 압축 파일 약 1.1GB를 내려받고 선택한 1,512장만 추출합니다. 세 모델 생성·평가에는 추가 디스크 공간과 긴 CPU 실행 시간이 필요합니다.

생성 파일은 다음과 같습니다.

```text
artifacts/rtdetr_fp32.onnx
artifacts/rtdetr_int8.onnx
artifacts/rtdetr_int4.onnx
runs/onnx_eval/<실행 ID>/metrics.json
runs/onnx_eval/<실행 ID>/class_metrics.csv
runs/onnx_eval/<실행 ID>/profile.json
runs/onnx_eval/<실행 ID>/provenance.json
```

각 ONNX 파일에는 모델 종류, 출처 체크포인트·데이터·설정 해시가 **파일 내부 메타데이터**로 기록됩니다. 별도 manifest 파일 없이 모델 한 개를 올려 재평가할 수 있습니다. `.manifest.json`은 생성 과정의 감사 기록입니다.

INT8은 `quantize_static`으로 Conv·MatMul의 **가중치와 활성값**을 8비트로 변환합니다. 선택한 512장으로 activation range를 보정합니다. 저장 결과는 QDQ ONNX이며, 평가 스크립트가 CPU 최적화 그래프에 정수 연산 노드가 실제 생성됐는지 확인합니다. 정수 커널이 없다면 평가를 실패시켜 FP32 실행을 INT8 성능으로 보고하지 않습니다.

INT4는 `MatMulNBitsQuantizer`의 블록 크기 32, 대칭 4비트 weight-only 형식을 사용합니다. **상수 가중치 MatMul**만 `MatMulNBits`가 됩니다. 일부 `Gemm`은 수학적으로 동등한 MatMul+Add로 변환하여 그 가중치도 대상에 포함합니다. Conv, 활성값, 동적 가중치 MatMul은 FP32입니다. 이 모델의 크기와 성능은 “RT-DETR 전체 INT4” 효과가 아닙니다. 런타임 그래프에서 `MatMulNBits`가 없으면 평가가 실패합니다. 이 범위는 [ONNX Runtime의 공식 INT4 지원 목록](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#quantize-to-int4uint4)에 따른 것입니다.

## 2. 이미 만든 모델 파일만 업로드해 평가

노트북의 데이터 준비 셀까지만 실행하고, 파일 업로드 셀에서 내려받아 둔 `.onnx`를 올린 다음 다음 명령을 실행할 수도 있습니다. FP32 기준도 같은 세션에서 평가해야 시간·메모리 비교가 가능합니다. 양자화 모델의 출처 체크포인트·데이터·설정 해시가 다르면 실행이 중단됩니다.

```python
from google.colab import files
uploaded = files.upload()  # ONNX 파일 하나
model_path = next(iter(uploaded))
subprocess.run([sys.executable, 'scripts/onnx_eval.py', '--model', model_path,
                '--run-id', 'uploaded_int4'], check=True)
```

모델 파일의 ONNX Runtime provider는 **CPUExecutionProvider**로 고정합니다. 같은 머신에서 세 모델을 별도 프로세스로 평가합니다. GPU 배정이 달라도 결과 조건이 CPU로 동일하게 유지되지만, 코랩 CPU 종류가 다른 세션 간 속도는 직접 비교하지 마세요. 로컬 분석기는 export 코드 버전이 다른 모델의 비교를 거부합니다. 같은 공식 모델·설정·export 코드에서 FP32 ONNX 파일의 직렬화 해시만 다른 경우에는 경고를 남깁니다.

## 3. 결과 ZIP을 로컬에서 분석

노트북 마지막 평가 셀이 FP32·INT8·INT4의 JSON/CSV를 ZIP 한 개로 다운로드합니다. 로컬에서 Python 표준 라이브러리만으로 비교할 수 있습니다.

```powershell
python .\scripts\analyze_real_ptq.py .\real_ptq_results_YYYYMMDD_HHMMSS.zip --output .\runs\real_ptq_analysis
```

`summary.csv`, `analysis.md`가 생성됩니다. 모델·데이터·평가 코드·ONNX Runtime이 다르면 합치지 않습니다. 결과의 ΔAP point는 `(FP32 mAP50:95 − 양자화 mAP50:95) × 100`으로, 양수는 정확도 손실입니다.

## 측정 항목과 해석

- **정확도:** COCO bbox mAP50:95, AP50, AP75, 80개 클래스별 AP.
- **실행:** batch 1, 640×640, 10장 warmup 후 평가 1,000장. 입력 준비·ONNX Runtime 호출·전체 처리의 mean/p50/p95와 관측 FPS.
- **메모리:** 모델 적재 뒤 RSS와 평가 중 최대 프로세스 RSS(20ms 표본). 매우 짧은 피크는 놓칠 수 있습니다.
- **크기·적용 범위:** ONNX 파일 바이트, 그래프의 QDQ/MatMulNBits 수, CPU 최적화 그래프의 정수 연산 수. 압축률은 FP32 ONNX 파일과 비교합니다.

INT8 정적 QDQ가 모든 지원 연산을 반드시 정수 커널로 바꾸는 것은 아닙니다. ONNX Runtime의 최적화 결과를 확인해 실제 적용 범위를 해석하세요. INT4는 활성값이 FP32인 **weight-only**라 INT8 W8A8과 양자화 범위가 다릅니다. 속도 향상을 보장하지 않습니다. [ONNX Runtime 공식 양자화 설명](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)이 정적 PTQ, QDQ, INT4 제한과 하드웨어에 따른 속도 차이를 설명합니다.
