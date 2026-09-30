# COCO 사전학습 RT-DETR R50 양자화 실험

**이번 프로젝트의 평가 환경은 Google Colab입니다.** 로컬에서 [`export_coco_artifact.py`](scripts/export_coco_artifact.py)로 양자화 모델 `.pth` 하나를 만든 뒤, [Colab에서 노트북 열기](https://colab.research.google.com/github/foxyaef/haiteam4_finetune/blob/main/colab/RTDETR_COCO_Eval.ipynb)를 눌러 파일을 올리면 같은 세션에서 FP32와 모델을 평가하고 결과 ZIP을 내려받습니다. 로컬에서 `python scripts/analyze_colab.py <다운로드한 ZIP>`으로 정확도·시간·메모리를 비교합니다. 설치, 명령, 지표 해석은 [코랩 실행 안내](docs/COLAB_WORKFLOW.md)에 있습니다. 현재 형식은 **fake PTQ 정확도 실험용**이며 실제 INT4 정수 가속을 측정하지 않습니다.

공식 **RT-DETR v1 R50-vd의 COCO 사전학습 모델을 그대로** 사용합니다. BDD100K 파인튜닝이나 체크포인트 재학습은 이번 실험에 없습니다. FP32와 구성요소별 W8A8·W6A6·W4A4의 **정확도 변화**를 같은 이미지에서 비교합니다. 현재 저비트 코드는 *fake quantization*이므로 실제 INT4 추론 속도나 모델 압축률을 보여주지는 않습니다.

## 처음 실행하는 순서

Windows PowerShell, Python 3.12 또는 3.13, Git 기준입니다. 설치 장치에 맞춰 `xpu`(Intel), `cu128`(NVIDIA), `cpu` 중 하나를 고르세요. 아래는 CPU 예시이며 RT-DETR R50의 1,000장 평가는 CPU에서 오래 걸릴 수 있습니다.

```powershell
git clone https://github.com/foxyaef/haiteam4_finetune.git
cd haiteam4_finetune
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Backend cpu
.\.venv\Scripts\python.exe .\prepare_data.py
.\coco.cmd check
.\coco.cmd run --group baseline --device cpu
```

`setup.ps1`은 고정된 공식 RT-DETR 소스와 **COCO 80클래스 사전학습 가중치**를 내려받아 해시를 확인합니다. 기본 Python 경로가 다르면 `-Python 'Python 실행 파일의 절대경로'`를 지정하세요. `prepare_data.py` **한 파일을 실행**하면 COCO 2017 val 이미지·어노테이션 아카이브를 받고, 필요한 이미지 1,512장만 추출해 calibration·평가 목록과 폴더를 만듭니다. 데이터 준비 파일은 Python 표준 라이브러리만 사용합니다. 중단되면 같은 명령을 다시 실행하고, 다운로드 없이 검사할 때는 `prepare_data.py --verify-only`를 사용하세요. [데이터 준비 상세](docs/COCO_DATA.md)

## 만들어지는 데이터 구조

```text
data/coco/
├─ downloads/                         원본 val2017.zip, annotations_trainval2017.zip
├─ val2017/                           고정 선택한 JPEG 1,512장
├─ annotations/
│  ├─ calibration.json                 val 512장
│  └─ evaluation.json                  val 1,000장
└─ manifest.json                       선택 목록·아카이브 해시

weights/rtdetr_r50vd_6x_coco_from_paddle.pth    공식 COCO 사전학습 가중치
runs/coco_phase1/                              조건별 실험 결과
```

Calibration 512장과 평가 1,000장은 **서로 겹치지 않는 COCO val2017 부분집합**입니다. 원본 모델은 COCO train에서 학습된 가중치이며 이번 프로젝트는 추가 학습을 하지 않습니다. 따라서 이 실험의 mAP는 **선택한 1,000장에 대한 값**이고, 공식 COCO val 5,000장 전체 점수와 직접 비교하면 안 됩니다. 팀원마다 같은 `data/coco/manifest.json`과 `upstream-lock.json`의 해시를 확인하세요.

## 1차 실험 실행

총 **16조건**: FP32 1개, 전체 양자화 3개, 네 구성요소 각각 8·6·4비트 12개입니다. 각 명령은 같은 COCO 사전학습 가중치에서 새로 시작합니다.

```powershell
.\coco.cmd run --group all --bits 8 --device cpu
.\coco.cmd run --group all --bits 6 --device cpu
.\coco.cmd run --group all --bits 4 --device cpu
.\coco.cmd run --group backbone --bits 4 --device cpu
.\coco.cmd run --group encoder --bits 4 --device cpu
.\coco.cmd run --group decoder --bits 4 --device cpu
.\coco.cmd run --group head --bits 4 --device cpu
.\coco.cmd report
```

각 구성요소의 8비트·6비트도 같은 방식으로 실행합니다. `--device`는 설치 백엔드에 맞게 `cpu`, `xpu`, `cuda` 중 지정하세요. 다섯 명의 역할 분담과 전체 명령은 [실험 프로토콜](docs/COCO_PROTOCOL.md)에 있습니다. 결과 폴더는 `runs/coco_phase1/<조건>/`이며 `metrics.json`(전체 AP), `class_metrics.csv`(80클래스별 AP), `provenance.json`(모델·데이터 해시, 실제 양자화 범위)을 보관합니다. `report`는 누락된 조건까지 `summary.csv`에 표시합니다.

주 지표는 **mAP50:95와 FP32 대비 ΔAP point**입니다. AP50, AP75, 클래스별 AP도 함께 봅니다. 가중치는 출력 채널별 대칭 MinMax, 활성값은 텐서별 비대칭 MinMax로 처리하고, 실제 호출된 `Conv2d`·`Linear`만 양자화합니다. Attention의 functional MatMul, 정규화, softmax, deformable sampling 등은 FP32에 남습니다. 이 범위를 벗어난 "전체 모델 INT4"라고 해석하지 마세요.

## 코랩에서 추가로 측정할 것

코랩 노트북은 같은 세션의 FP32와 양자화 조건에 대해 **추론시간 p50·p95, 관측 FPS, 최대 프로세스 RSS, CUDA 최대 할당·예약 메모리**를 기록합니다. 코랩 GPU와 PyTorch 버전도 함께 저장해 비교 조건을 확인합니다. 시간·메모리는 이 fake PTQ 구현을 코랩에서 실행한 값입니다.

[RT-DETR 내부 구조](docs/RTDETR_STRUCTURE.md)에는 Backbone·Hybrid Encoder·Decoder·Head의 연산 흐름과 양자화 경계를 설명했습니다. 기존 BDD100K 파인튜닝 파일(`download_bdd100k.py`, `run.cmd`, `phase1.cmd` 등)은 이전 실험 재현을 위해 남겨두었지만 **이번 COCO 실험의 실행 경로가 아닙니다**. 데이터·가중치는 Git에 포함되지 않으며 [COCO 이용 조건](https://cocodataset.org/#termsofuse)을 확인하세요.
