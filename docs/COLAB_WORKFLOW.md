# Colab 공통 평가 안내

[평가 노트북](https://colab.research.google.com/github/foxyaef/haiteam4_finetune/blob/main/colab/RTDETR_COCO_Eval.ipynb)은 로컬에서 만든 ONNX 모델을 받습니다. Colab에서 양자화하지 않습니다. 노트북은 저장소·공식 RT-DETR 체크포인트·COCO val2017 부분집합을 내려받고 FP32 기준 ONNX를 만든 뒤, 업로드 모델과 동일한 조건으로 평가합니다. 처음 준비할 때 데이터 ZIP 약 1.1GB와 추가 디스크 공간이 필요합니다. CPU 평가 1,000장은 시간이 오래 걸릴 수 있습니다.

## 모델 제출 조건

- 공식 COCO 사전학습 RT-DETR v1 R50-vd에서 출발한 모델. 다른 체크포인트·클래스 매핑·후처리라면 비교 대상에서 제외합니다.
- 외부 가중치 파일이 없는 **단일 ONNX**. Colab의 ONNX Runtime CPU에서 실행 가능해야 합니다.
- 입력: `images` float32 [1,3,640,640], `orig_target_sizes` int64 [1,2].
- 출력 순서: `labels` [1,300], `boxes` [1,300,4], `scores` [1,300]. COCO 80개 클래스 인덱스와 원본 이미지 픽셀 좌표 xyxy.
- 평가 1,000장은 calibration·방법 선택에 사용하지 않습니다.

프로젝트의 `onnx_real_ptq.py`가 만든 파일은 내장 출처 해시를 검사합니다. 다른 도구가 만든 ONNX는 업로드 시 `register_model.py`가 입출력 형식과 CPU 샘플 실행을 확인하고 출처·기법을 **팀 선언**으로 기록합니다. 이 검사는 모델의 학습 출처나 양자화 비트를 증명하지 않습니다. 잘못된 클래스 매핑이나 박스 해석은 자동으로 모두 발견할 수 없으니 모델 생성 기록으로 확인하세요.

## 실행과 결과

1. 노트북 셀을 순서대로 실행합니다. `method_name`에 실험 이름을 적고 ONNX 파일 **한 개**를 업로드합니다.
2. 평가기는 FP32와 업로드 모델 각각을 별도 프로세스에서 실행합니다. ONNX Runtime `CPUExecutionProvider`, batch 1, 입력 640×640, warmup 10장, COCO val2017 고정 평가 1,000장입니다.
3. `coco_eval_results_*.zip`을 내려받습니다. 로컬 저장소에서 다음 명령으로 분석합니다.

```powershell
python scripts/analyze_real_ptq.py .\coco_eval_results_YYYYMMDD_HHMMSS.zip
```

ZIP에는 모델별 `metrics.json`, `class_metrics.csv`, `profile.json`, `provenance.json`이 있습니다. 분석 결과는 `runs/real_ptq_analysis/summary.csv`와 `analysis.md`입니다.

- **정확도:** bbox mAP50:95, AP50, AP75, 클래스별 AP. FP32 대비 ΔAP point = (FP32 AP − 실험 AP) × 100.
- **속도:** ONNX Runtime 호출 mean/p50/p95, 전체 처리 시간과 관측 FPS. 같은 Colab 세션에서만 속도 비율을 비교하세요.
- **메모리·크기:** 모델 적재 뒤 RSS, 실행 중 표본화한 최대 프로세스 RSS, ONNX 파일 크기. 20ms보다 짧은 피크는 놓칠 수 있습니다.
- **실제 적용 범위:** ONNX 그래프와 CPU 최적화 그래프의 연산 수. 파일 이름이나 `method_name`은 적용 범위의 증거가 아닙니다.

COCO val2017 전체 5,000장 공식 mAP와 이 **1,000장 부분집합 mAP**를 직접 비교하지 마세요. 파일 내부의 출처 해시가 다르면 평가가 중단됩니다. 다른 도구 모델의 출처는 팀 선언이므로 등록 전에 직접 검증해야 합니다. 로컬과 Colab의 FP32 ONNX 파일 해시가 다르면 분석기에 경고가 남습니다.

제공된 INT8 예제는 calibration을 쓰는 Conv·MatMul 정적 QDQ입니다. 제공된 INT4 예제는 상수 가중치 MatMul의 packed weight-only이며 Conv·활성값은 FP32입니다. [ONNX Runtime INT4 지원 범위](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#quantize-to-int4uint4)를 참고하세요. 팀의 별도 방법이 전체 INT4인지 여부는 제출 모델의 실제 연산과 저장형식을 확인한 뒤 판단합니다.
