# Colab 수정본 검증 기록

검증일: 2026-10-04 (한국 시간). 기준 GitHub 버전은 `6ddf0a7`입니다.

## 수정한 문제

- 기존 Colab 패키지와 충돌할 수 있는 설치를 평가 전용 Python 환경으로 분리했습니다. CUDA/cuDNN 의존성을 설치하고 평가 프로세스에서 preload합니다.
- INT4의 DequantizeLinear를 INT8 증거로도 표시하던 오류를 수정했습니다.
- 모델 하나의 실패로 전체 비교와 ZIP 다운로드가 중단되던 흐름을 수정했습니다. 성공 결과·실패 상태·로그를 함께 보관합니다.
- 3장 실행 확인 모드를 추가하고, 선택한 이미지 ID만 대상으로 COCO AP를 계산합니다. 워밍업보다 이미지 수가 적어도 시간 통계가 비어 있지 않습니다.
- 지연 반복 시 예측을 중복 제출하지 않으며, AP에는 이미지당 첫 예측만 사용합니다.
- 그래프 검사와 provider profiling을 RAM/지연 측정에서 분리했습니다. GPU 메모리를 현재 프로세스에 귀속할 수 없으면 null과 이유를 기록합니다.
- 별도 profiling에서 실제 CPU/CUDA kernel event를 기록하고, CUDA FP32 비교에서는 TF32를 끕니다.
- 수정한 평가 코드를 노트북에 포함하고 SHA256을 검증합니다. 다운로드한 노트북이 기존 GitHub 코드를 다시 불러오는 문제를 방지합니다.

## 실제 확인 범위

| 확인 항목 | 결과 |
| --- | --- |
| nbformat 5.10.4 노트북 스키마 검사 | 8개 셀 유효 |
| 모든 Python 코드셀 문법·내장 코드 해시·저장소 파일 일치 | 통과 |
| 자동 테스트 | 10개 통과 |
| 새 Python 평가 환경 생성·CPU 의존성 설치·pip check | 통과 |
| 공통 COCO ZIP SHA256·1,512장 CRC·512/1,000 목록·겹침 검사 | 통과 |
| 노트북 코드셀 실행: 준비 → 모델 선택 → 평가 → 표·미리보기·ZIP | 통과, Colab UI 호출만 로컬 대체 |
| 실패한 모델 이후 평가 지속·성공 결과와 오류 로그 ZIP | 테스트 통과 |
| 업로드 취소 후 작업 폴더 복원 | 테스트 통과 |

CPU 실행에는 Python 3.12.14, ONNX 1.18.0, ONNX Runtime 1.22.1, pycocotools 2.0.10을 사용했습니다. 노트북 내장 평가 코드 ZIP의 SHA256은 `a6f4c1c5139cefb390305ffc02754a7e3c609d52b25d622bf37ceda14586c801`입니다.

## 사용한 실제 모델

아래 모델마다 공통 held-out 평가 이미지 3장을 실제 추론했습니다. 모든 모델에서 COCO 지표 계산, RAM/지연 측정, provider profiling, 예측 JSON, 클래스별 CSV, 탐지 미리보기와 ZIP 생성을 확인했습니다.

| 모델 | 이미지 수 | 결과 |
| --- | --- | --- |
| fp32.onnx | 3 | 통과 |
| encoder_transformer_w8a8.onnx | 3 | 통과 |
| encoder_transformer_w4a16.onnx | 3 | 통과 |
| encoder_cnn_w8a8.onnx | 3 | 통과 |
| encoder_cnn_w4a16.onnx | 3 | 통과 |
| encoder_both_w8a8.onnx | 3 | 통과 |
| encoder_both_w4a16.onnx | 3 | 통과 |

W8A8은 INT8 증거가, W4A16은 실제 INT4 initializer 증거가 확인됐습니다. W4A16의 DQ 연산만으로 INT8 증거를 표시하지 않습니다.

## 남아 있는 검증 범위

**실제 Colab 브라우저의 업로드·다운로드 UI, Colab GPU/CUDA 실행, 평가 1,000장 전체 실행은 이 검증에서 수행하지 않았습니다.** 로컬에서는 Google Colab UI 호출을 대체하고 나머지 노트북 코드·새 설치 환경·실제 모델 실행을 확인했습니다. GPU 의존성·preload·TF32 설정은 공식 ORT 문서에 맞췄으며, GPU 런타임별 지원 여부와 CPU fallback은 실제 실행 결과의 로그와 `provider_placement.json`으로 확인해야 합니다.

3장 결과의 AP·지연·메모리 수치는 연구 결론이나 최종 성능 비교에 사용하지 않습니다. Colab에서 `evaluation_mode=full`로 같은 1,000장을 평가하세요.
