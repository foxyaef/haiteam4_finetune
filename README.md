# RT-DETR-R50 / BDD100K 로컬 파인튜닝

이 프로젝트는 **공식 RT-DETR v1 R50-vd**, **COCO 전용 공식 pretrained weight**, **640×640 FP32**, **BDD100K Detection 공식 train/val**을 사용합니다. RT-DETRv2, R50-m, HGNet 모델이 아닙니다.

현재 연결한 데이터는 Berkeley 공식 서버의 `bdd100k_images_100k.zip` 및 `bdd100k_labels.zip` **원본 Labels 배포본**입니다. 현재 공식 “Detection 2020 Labels” 버튼은 다른 이미지 ZIP으로 잘못 연결되어 있어 사용하지 않았습니다. 이 데이터가 Detection 2020 재라벨링 배포본과 같다고 가정하지 마세요. 비교 실험에서는 이 프로젝트의 원본 ZIP 해시·변환 결과 JSON을 동일하게 공유해야 합니다. 출처와 변환 내역은 `reports/dataset_download_provenance.json`에 기록됩니다.

## 바로 시작하기

PowerShell에서 이 폴더를 연 다음 실행합니다. 가상환경을 활성화할 필요는 없습니다.

```powershell
.\run.cmd check
.\run.cmd smoke
```

현재 BDD100K 다운로드·변환이 완료되어 경로 수정이나 `prepare` 재실행 없이 학습을 시작할 수 있습니다. 최종 점검 결과는 `reports/dataset_ready.json`에서 확인하세요.

```powershell
.\run.cmd train
.\run.cmd evaluate --save-predictions
.\run.cmd benchmark
```

데이터가 없으면 학습이 시작되지 않습니다. `smoke`는 합성 입력으로 공식 가중치 로딩, 640×640 forward, loss, backward, optimizer update, inference를 검증합니다. 실제 BDD100K 성능이나 파인튜닝 결과가 아닙니다.

중단한 학습은 마지막 **완료한 epoch**부터 재개합니다. 진행 중이던 epoch의 미완료 batch는 다시 실행됩니다.

```powershell
.\run.cmd train --resume runs/fp32/last.pth
```

재개 시 config와 데이터 manifest가 원래 실험과 같아야 합니다. 새 실험은 YAML의 `output`을 새 폴더로 지정합니다. 원래 run을 덮어쓰지 않습니다.

## 데이터 준비

[BDD100K 공식 다운로드](http://bdd-data.berkeley.edu/download.html)의 **100K Images**와 **Labels** 원본 배포본을 사용합니다. 이 작업에서는 해당 파일을 이미 다운로드했습니다. **Detection 2020 Labels**는 별도 배포본이며 현재 버튼의 잘못된 연결 때문에 사용하지 않았습니다. 동영상, segmentation, tracking 데이터는 이 환경에 필요하지 않습니다.

압축을 풀어 기본 설정과 맞추거나 YAML에 절대 경로를 입력하세요. 다른 드라이브에 데이터를 보관해도 됩니다.

```text
data/bdd100k/
  images/100k/train/*.jpg
  images/100k/val/*.jpg
  labels/official_raw/det_train.json
  labels/official_raw/det_val.json
```

배포본에 따라 라벨 이름이 `det_v2_train_release.json`, `det_v2_val_release.json` 또는 `bdd100k_labels_images_train.json`, `bdd100k_labels_images_val.json`일 수 있습니다. 파일 이름을 추측해 바꾸기보다 YAML의 `train_labels`, `val_labels`를 실제 파일로 지정하세요. 비교 실험 참여자 모두 **같은 라벨 release와 생성된 COCO JSON**을 공유해야 합니다.

`prepare`는 이미지 존재·손상 검사, 중복 이름·train/val 겹침 검사, 원본 이미지 크기 확인, 2D box 변환, 클래스 매핑, 라벨 및 결과 JSON의 SHA256 기록을 수행합니다. 제공된 공식 라벨 파일의 모든 프레임을 유지하며 랜덤 split을 만들지 않습니다. 배포본 차이를 위해 train 69,863 또는 70,000장, val 10,000장을 허용합니다. 수량만으로 공식 배포본임을 인증할 수 없으므로 원본 다운로드 출처도 보관하세요. 이미지 바이트 전체 해시는 이번 압축 해제 단계에서 별도로 기록했습니다.

박스는 설정의 `bbox_coordinate_convention: scalabel_inclusive`에 따라 `width=x2-x1+1`, `height=y2-y1+1`로 해석한 뒤 이미지 경계로 clip합니다. 이는 [Scalabel 공식 Box2D 변환](https://github.com/scalabel/scalabel/blob/master/scalabel/label/transforms.py)을 따르며, 끝점이 같은 1픽셀 폭/높이의 박스도 유지합니다. 반전·비정상 숫자 박스는 오류로 중단합니다. 이미지와 겹치는 면적이 없는 박스는 `*.filtered_boxes.json`에 원본 좌표·이미지·ID·사유를 기록하고 제외하며, 해당 이미지 자체는 유지합니다. 현재 원본에서는 train의 `9f68b0a6-b2c427e6.jpg` 객체 ID 4가 오른쪽 경계 바깥 x=1280에만 존재하는 이 경우에 해당합니다. 잘린/가려진 객체는 유지합니다. `crowd`/`ignored`는 COCO `iscrowd`로 변환합니다. 클래스에 없는 2D box는 조용히 버리지 않고 오류로 중단합니다. 이전 환경 설치 때의 연속 XYXY 가정은 실제 데이터 확인 후 수정했습니다.

원본 per-image JSON의 `frames[0].objects` 중 `box2d`가 있는 객체만 Detection `labels`로 모았습니다. 원본 좌표·속성·train/val 분할을 보존하고 이미지 이름에 `.jpg`만 붙입니다. 원본 ZIP은 `data/downloads`에 그대로 남깁니다. 압축 해제 시 CRC와 이미지별 SHA256을 기록하며 `data/image_sha256.jsonl`로 확인할 수 있습니다.

네트워크가 끊기면 다음 명령으로 조각 중간부터 이어받습니다. 완료된 ZIP도 해시를 검사하고 재사용합니다.

이어받기 조각과 조립 중 파일은 OneDrive 동기화 충돌을 피하도록 Windows 임시 폴더의 `rtdetr_bdd100k_download_cache`에 보관합니다. 최종 ZIP만 `data/downloads`에 복사합니다. 다른 임시 위치를 원하면 `BDD100K_DOWNLOAD_CACHE` 환경 변수로 지정할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe scripts/download_bdd.py http://128.32.162.150/bdd100k/bdd100k_labels.zip http://128.32.162.150/bdd100k/bdd100k_images_100k.zip
```

다운로드만 완료된 경우에는 순서대로 처리합니다.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_bdd_downloads.py labels
.\.venv\Scripts\python.exe scripts/prepare_bdd_downloads.py images
.\run.cmd prepare
```

현재 작업 상태는 `data/downloads/download-status.json`, 압축 해제 상태는 `reports/dataset-extraction-status.json`에 기록됩니다. Test 20,000장은 원본 이미지 ZIP에 보관하며 학습·검증 폴더에는 추출하지 않습니다.

| COCO ID | 클래스 | 허용하는 원본 별칭 | 내부 학습 ID |
|---:|---|---|---:|
| 1 | pedestrian | person | 0 |
| 2 | rider | | 1 |
| 3 | car | | 2 |
| 4 | truck | | 3 |
| 5 | bus | | 4 |
| 6 | train | | 5 |
| 7 | motorcycle | motor | 6 |
| 8 | bicycle | bike | 7 |
| 9 | traffic light | | 8 |
| 10 | traffic sign | | 9 |

원본 공식 가중치에서 backbone/encoder/decoder와 호환되는 box head를 불러옵니다. 80→10 클래스 변경으로 분류 head와 denoising class embedding만 초기화합니다. 초기화한 key 목록을 `initial_load.json`에 저장합니다.

## 기본 학습 설정

- 입력: RGB, 640×640 bilinear 직접 resize, [0,1] scaling. Letterbox와 ImageNet normalization 없음. Train/val/PTQ에서 같은 전처리를 사용하세요.
- FP32; AMP/TF32 비활성화, EMA 없음. backbone의 원래 frozen normalization/stem 정책 유지.
- Batch 1, gradient accumulation 8, workers 0. BatchNorm 통계는 실효 배치 8의 통계와 같지 않습니다.
- AdamW, LR 1e-4, backbone LR 1e-5, weight decay 1e-4.
- 50 epochs, 1 epoch linear warmup + cosine decay, gradient clipping 0.1.
- 좌우 반전 확률 0.5. Multi-scale 비활성화로 640 고정.
- Early stopping: validation mAP50:95가 10 epoch 동안 개선되지 않을 때.

현재 PC용 설정은 Intel Arc 130V(XPU), RAM 16GB를 전제로 배치 1부터 시작합니다. 전체 BDD100K 파인튜닝은 상당한 시간이 필요할 수 있습니다. 검증한 합성 1-step 시간은 `runs/smoke/smoke_result.json`에 기록됩니다. 실제 객체 밀도가 높은 이미지의 메모리와 처리 시간은 다릅니다. 메모리 부족 시 먼저 accumulation은 유지하고 batch 1/workers 0인지 확인하세요. 장시간 학습 시 절전 설정과 전원을 확인하세요.

CPU 실행은 `--device cpu`로 명시적으로 선택할 수 있습니다. NVIDIA PC로 옮기면 `setup.ps1 -Backend cu128`로 해당 빌드를 설치하고 YAML의 device를 `cuda`로 설정합니다. 가상환경은 복사하지 말고 새 위치에서 다시 만드세요.

## 지표와 저장 파일

| 항목 | 정의 / 위치 |
|---|---|
| 주 지표 | pycocotools bbox mAP@0.50:0.05:0.95, area=all, maxDets=100 |
| best.pth | 전체 공식 val의 주 지표가 엄격히 개선될 때 저장한 raw FP32 모델 |
| last.pth | 마지막 완료 epoch 모델, optimizer, best score, resume 상태 |
| mAP50 | COCO AP@0.50 |
| Precision / Recall | confidence ≥0.5, IoU=0.5, COCO matching 및 crowd/ignore 규칙, micro 집계 |
| Class-wise | AP, AP50, AR100, threshold Precision/Recall, TP/FP/FN/GT |
| loss | train total(보조·denoising 포함), train main, val main(최종 출력만) |
| history.csv | epoch별 학습·검증 loss와 성능 |
| epoch_XXX/ | 각 epoch의 metrics.json, class_metrics.csv |
| environment.json | seed/config, repository commit, 라이브러리 버전, 로컬 코드 해시, 초기 가중치·데이터 해시 |
| *.sha256.json | 저장한 checkpoint의 해시 |
| evaluation/ | best 모델 재평가 지표 및 선택적으로 COCO predictions.json |
| benchmark/efficiency.json | FP32 크기, latency, FPS, peak 메모리와 측정 조건 |

모든 AP/Precision/Recall은 **0–1** 단위입니다. 백분율 표시는 100을 곱하세요. GT가 없는 클래스의 AP/Recall은 `null`이며, COCO는 그 클래스를 mAP 평균에서 제외합니다. `train` 클래스가 특정 split에 없더라도 모델의 10개 출력과 클래스 매핑은 유지됩니다.

AP 계산에는 confidence 0.5 필터를 적용하지 않습니다. 공식 postprocessor의 top-300 예측을 전달하고 COCO maxDets=100을 적용합니다. Precision은 오탐률 자체가 아니며 `TP/(TP+FP)`입니다. threshold는 모델 간 동일하게 고정해야 합니다. Val loss는 main loss로 train total과 항목 수가 다릅니다. 수렴 비교에는 train main도 함께 보세요.

벤치마크는 batch 1, warmup 10회, 측정 50회, 동일 장치에서 FP32 model+postprocess를 측정합니다. 입력은 미리 장치에 올린 합성 텐서이며 파일 읽기/디코딩/resize/장치 전송은 제외합니다. 평균·p50·p95 latency, 평균 시간의 역수 FPS, checkpoint 파일 크기 및 tensor bytes, 장치 allocator peak, 10ms 간격으로 샘플링한 process RSS peak를 구분합니다. 장치 메모리와 RSS를 단순 합산하면 공유 메모리가 중복될 수 있습니다.

## PTQ 후속 실험 준비 범위

이번 환경의 실행 범위는 **FP32 파인튜닝·평가·벤치마크**입니다. Basic/MSE/Clipping/BRECQ/QDrop 및 INT8/INT6/INT4·mixed precision 알고리즘은 아직 구현하지 않았습니다. 실험 조건은 `PTQ_PLAN.md`에 정리했습니다. 이름만 붙인 가짜 BRECQ/QDrop 구현이나 fake quant의 시간을 정수 추론 성능으로 보고하지 않습니다.

`prepare`는 train에서 seed 42로 512장을 고정 선택해 `data/calibration_manifest.json`과 `data/annotations/calibration.json`을 만듭니다. Val에서는 절대 선택하지 않습니다. 이 subset은 PTQ용으로 별도 관리하되 FP32 학습의 공식 train 전체에는 포함됩니다. 학습에서 제외한 holdout을 원하면 모든 비교 실험의 프로토콜을 먼저 함께 변경해야 합니다.

```powershell
.\run.cmd module-map
```

이 명령은 중복 없는 backbone / encoder / decoder / head 파라미터 분류를 저장합니다. RT-DETR에서 head는 독립적인 최상위 모듈이 아니라 decoder 아래에 있으므로 head 분류를 우선 적용합니다.

## 재설치 및 검증

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\run.cmd check
.\run.cmd smoke
```

Python 3.12, torch 2.8.0, torchvision 0.23.0을 사용합니다. `requirements.txt`는 직접 의존성을 고정하고, 설치된 전체 버전은 `reports/installed-packages.txt` 및 실행별 environment에 보관합니다. `setup.ps1 -Python '다른 Python 3.12 경로'`로 interpreter 경로를 지정할 수 있습니다.

공식 저장소는 `vendor/RT-DETR`에 원본 그대로 보관하고 commit을 고정합니다. 공식 소스의 오래된 torchvision augmentation과 사용하지 않는 RegNet import를 실행하지 않도록 `scripts/official.py`가 필요한 공식 모듈만 로딩합니다. 모델/criterion 소스는 변경하지 않았습니다. 로컬 학습 루프와 데이터 전처리는 `scripts/`에 분리되어 있습니다. upstream `tools/train.py`는 이 최신 XPU 환경의 실행 진입점이 아닙니다.

seed와 epoch별 sampler 상태를 고정하지만 장치별 grid sampling backward 등은 완전 결정적이지 않을 수 있습니다. 동일 seed가 서로 다른 하드웨어에서 bitwise 동일 결과를 보장하지는 않습니다.

설치 시 생성된 `runs/smoke`, `runs/integration_test`, `runs/integration_fixture`는 모두 합성 입력 검증 기록입니다. 실제 BDD100K 체크포인트로 사용하지 마세요. 실제 실험의 기본 출력은 별도 `runs/fp32`입니다. 검증 중 발견해 수정한 저장 오류의 이전 기록은 `runs/integration_failed_serialization`에 보관했습니다.

출처: [공식 RT-DETR PyTorch 모델/가중치](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch), [BDD100K 클래스 형식](https://github.com/bdd100k/bdd100k/blob/master/doc/source/format.rst), [PyTorch Intel GPU 안내](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html).
