# 로컬 양자화 안내

공통 목표는 **Colab 평가기에 제출할 단일 ONNX 모델**을 로컬에서 만드는 것입니다. 양자화 기법 선택·구현은 팀원별로 달라도 됩니다. 같은 [FP32 기준 모델](../README.md#1-로컬-양자화-모델-만들기)과 COCO 512장 calibration 목록을 사용하고, 원본 모델·기법·적용 연산·라이브러리 버전을 실험 노트에 남기세요.

## 공통 준비

README의 설치·소스/체크포인트/데이터 준비·FP32 export를 완료합니다. `python prepare_data.py --verify-only`가 통과해야 합니다. `data/coco/annotations/calibration.json`에 512장 목록이, `evaluation.json`에 겹치지 않는 1,000장 목록이 있습니다. **평가 1,000장을 양자화 파라미터 선택이나 튜닝에 사용하지 마세요.**

## 제공하는 예제 PTQ

```powershell
python scripts/onnx_real_ptq.py quantize --source artifacts/rtdetr_fp32.onnx --precision int8 --output artifacts/rtdetr_int8.onnx
python scripts/onnx_real_ptq.py quantize --source artifacts/rtdetr_fp32.onnx --precision int4 --output artifacts/rtdetr_int4.onnx
```

INT8은 calibration 512장으로 Conv·MatMul의 가중치와 활성값을 정적 QDQ 양자화합니다. INT4는 calibration 없이 상수 가중치 MatMul을 packed 4비트로 바꾸는 **weight-only** 예제입니다. Conv와 활성값은 FP32로 남습니다. 따라서 두 예제는 같은 범위의 양자화 실험이 아니며, INT4 예제를 전체 W4A4라고 발표하면 안 됩니다.

## 팀원이 별도 도구로 만든 ONNX

모델은 **파일 하나**에 모든 가중치가 들어 있어야 합니다. 입력은 `images`: float32 [1,3,640,640], `orig_target_sizes`: int64 [1,2]이고 출력 순서는 `labels`, `boxes`, `scores`여야 합니다. 모두 batch 1, 예측 300개입니다. 클래스는 COCO 80개 인덱스 0–79, 박스는 원본 이미지 픽셀 기준 xyxy여야 합니다. 다른 출력 형식이면 공식 deploy postprocessor를 포함해 다시 ONNX로 내보내세요.

출처 메타데이터가 없는 모델은 로컬에서 등록할 수 있습니다.

```powershell
python scripts/register_model.py --model artifacts/my_quant.onnx --baseline artifacts/rtdetr_fp32.onnx --output artifacts/my_registered.onnx --method "my_method"
```

등록기는 ONNX 유효성, 단일 파일 여부, 입출력 이름 및 샘플 실행 결과 형식을 확인합니다. 원본 체크포인트·평가 프로토콜 메타데이터는 기준 모델에서 복사하고 **출처를 `team_declared`로 표시**합니다. 이는 모델이 실제로 같은 체크포인트에서 왔는지, 이름대로 양자화되었는지를 증명하지 않습니다. 등록 전 원본·방법·가중치 저장형식·추론 연산을 팀에서 검증하세요. 등록하지 않은 모델도 Colab 업로드 셀에서 동일한 방식으로 등록됩니다.

Colab에는 `my_registered.onnx` 또는 예제 양자화 ONNX 파일 **한 개**를 업로드합니다. 동일한 실험의 FP32 결과도 같은 Colab 세션에서 자동 생성·평가합니다.
