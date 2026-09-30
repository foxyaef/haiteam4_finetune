# COCO RT-DETR 코랩 평가: 모델 파일 하나만 올리는 흐름

이 흐름은 공식 COCO 사전학습 RT-DETR v1 R50-vd를 기준으로 고정한 1,000장의 COCO val2017 부분집합에서 정확도와 **코랩 런타임의** 자원 사용량을 비교합니다. 이 프로젝트의 평가 환경은 Google Colab입니다.

## 1. 로컬에서 양자화 모델 파일 만들기

프로젝트 폴더에서 COCO 데이터·공식 모델 준비가 끝난 뒤 실행합니다. Windows에서는 기존 `setup.ps1`, `prepare_data.py`를 먼저 실행하세요.

```powershell
.\.venv\Scripts\python.exe .\scripts\export_coco_artifact.py --group backbone --bits 4 --device cpu --output .\artifacts\backbone_w4a4.pth
```

`group`은 `all`, `backbone`, `encoder`, `decoder`, `head`; `bits`는 8, 6, 4입니다. 다른 팀원의 조건도 각각 파일 하나로 내보냅니다. Export 시 512장 calibration을 한 번 수행하며, 파일에는 양자화된 정수 가중치, 채널별 scale, 활성값 범위, 나머지 FP32 가중치와 실험 식별 해시가 들어갑니다. `.pth`는 `torch.int8`에 4/6/8비트 값을 담습니다. **압축된 INT4/INT6 파일이 아니며**, 로드 뒤 FP32로 역양자화해 fake quantization 추론을 합니다.

현재 지원하는 업로드 형식은 위 exporter가 만든 `rtdetr-coco-fake-ptq-v1` 파일입니다. BRECQ, QDrop, QRT-DETR 등 다른 방법을 평가하려면 해당 방법이 동일한 artifact 형식으로 출력하거나 별도 로더를 구현해야 합니다. 임의의 체크포인트·ONNX 파일을 그대로 업로드하면 동작하지 않습니다.

## 2. 코랩에서 실행

[평가 노트북](../colab/RTDETR_COCO_Eval.ipynb)을 Google Colab에서 열고, 가능하면 GPU 런타임으로 설정한 뒤 위에서 아래로 실행합니다. 노트북이 공개 저장소, 고정된 공식 소스·FP32 가중치, COCO val2017 자료를 내려받습니다. 데이터 압축 파일 다운로드가 약 1GB 이상이고 저장 공간도 추가로 필요합니다. `requirements.txt`를 설치하며 코랩의 PyTorch 버전과 GPU 종류는 실행 결과에 기록합니다.

업로드 셀에서 **export한 `.pth` 하나**를 올립니다. 큰 파일 업로드가 불편하면 Google Drive를 마운트하고 `artifact_path`를 그 파일의 경로로 바꾸세요. 노트북은 같은 세션에서 FP32와 업로드 모델을 각각 1,000장으로 평가하고 ZIP 한 개를 내려받습니다. 새 세션마다 설치·데이터 다운로드가 다시 필요할 수 있습니다. 같은 런타임에서 다른 모델을 추가 평가하려면 업로드·평가·ZIP 셀을 다시 실행하세요.

## 3. 로컬에서 ZIP 분석

ZIP을 프로젝트 폴더로 옮긴 후 별도 라이브러리 설치 없이 실행할 수 있습니다.

```powershell
python .\scripts\analyze_colab.py .\colab_results_YYYYMMDD_HHMMSS.zip --output .\runs\colab_analysis
```

여러 ZIP 비교도 가능합니다. 단, **FP32 기준 결과는 정확히 하나**만 넘기세요. 여러 팀원의 ZIP에는 각각 기준 결과가 들어 있으므로, 하나의 ZIP에서는 baseline+모델을 그대로 사용하고 다른 ZIP에서는 artifact 폴더만 추출해 추가 인자로 주면 됩니다. 모델/데이터/평가 설정/평가 코드 해시가 다른 결과는 합치지 않습니다. 서로 다른 GPU나 PyTorch 버전이면 속도 비교에 주의 표시가 붙습니다. 출력은 `summary.csv`와 `analysis.md`입니다.

## 측정 결과의 범위

- 정확도: mAP50:95, AP50, AP75, 클래스별 AP. FP32 대비 ΔAP point는 `(FP32 mAP - 실험 mAP) × 100`입니다.
- 시간: batch 1, 입력 640×640, 10장 워밍업 후 1,000장. 이미지 준비, 전송, 모델, 후처리, 전체 시간의 mean/p50/p95와 Python 처리 포함 관측 FPS입니다. CUDA 동기화 후 각 구간을 잽니다.
- 메모리: 프로세스 RSS 최대값(20ms 표본), 모델 로드 후 RSS, CUDA peak allocated/reserved, 시스템 RAM. GPU 전체 장치의 다른 프로세스 메모리는 포함하지 않습니다.
- `artifact_bytes`는 업로드 파일 크기입니다. 포함된 FP32 가중치와 int8 컨테이너 때문에 INT4 배포 크기로 해석할 수 없습니다.

코랩의 하드웨어 배정은 세션마다 달라질 수 있습니다. 속도·메모리는 같은 세션에서 비교하고, 가능하면 GPU 종류가 같은 결과끼리 모으세요.
