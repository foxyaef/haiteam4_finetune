# 후속 PTQ 실험 프로토콜 (아직 실행 구현 없음)

1. 모든 방법은 동일 `best.pth`와 SHA256, 같은 calibration manifest, 전처리, 전체 validation annotation 및 `scripts/metrics.py`를 사용한다.
2. 비교 순서: FP32 → Basic → MSE/Clipping → BRECQ → QDrop. 각 방법을 INT8 → INT6 → INT4 순서로 평가한다. 비트 폭은 weight/activation을 구분해서 W8A8, W6A6, W4A4 등으로 명시한다. weight-only와 W/A 양자화를 같은 실험으로 취급하지 않는다.
3. scale granularity, symmetric/asymmetric, zero point, clipping 범위, rounding, calibration batch 수, reconstruction step/LR, QDrop probability, quantization 제외 연산을 config에 기록한다. Validation을 calibration이나 reconstruction에 사용하지 않는다.
4. 민감도: 한 component만 양자화하고 나머지는 FP32로 유지해 mAP50:95 저하량을 비교한다. Backbone, hybrid encoder, transformer decoder, classification/bbox head를 중복 없이 구분한다. 최상위 decoder 전체를 양자화한 뒤 head를 다시 별도 계산하면 중복된다. `module-map` 결과를 기준으로 한다.
5. INT4/INT8 mixed precision 후보를 validation으로 고르면 그 결과는 validation 기반 모델 선택 결과다. 독립 test 성능이라고 표현하지 않는다. 선택 정책과 탐색 후보/횟수를 기록한다.
6. 정수 하드웨어 경로가 없는 INT6/INT4는 fake-quant 정확도 실험으로 표시한다. FP32 저장 tensor의 파일 크기나 fake-quant latency를 실제 INT4 배포 크기·속도로 해석하지 않는다. packed artifact와 지원 backend가 있는 경우에만 실제 INT latency/FPS/peak memory를 측정한다.
7. FP32와 INT8/INT4 backend의 측정 범위를 맞춘다. 같은 장치, batch 1, 640×640, warmup, 동기화, postprocess 포함 여부를 고정한다. CPU와 XPU의 시간을 직접 양자화 이득으로 비교하지 않는다.

진행 상태: calibration 생성기, FP32 evaluator, module map, FP32 efficiency benchmark 준비. 실제 PTQ integration, reconstruction, backend export, mixed precision 검색은 별도 구현·검증이 필요하다.
