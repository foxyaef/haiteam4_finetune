# RT-DETR 공통 실험 환경

이 저장소는 팀이 **같은 모델·데이터·평가 지표**를 사용하도록 만드는 최소 환경입니다. 양자화 방법과 추론 코드는 각자 구현합니다. 이 저장소에는 INT4/INT8 양자화 예제나 특정 추론 엔진용 실행 코드가 없습니다.

## 공통 준비

Python 3.11 이상과 Git이 필요합니다. 아래 명령은 로컬 PC 또는 Colab에서 실행할 수 있습니다. 평가 지표 계산에는 `requirements.txt`가 필요하지만, 모델·데이터 다운로드 자체는 Python 표준 라이브러리만 사용합니다.

```powershell
git clone https://github.com/foxyaef/haiteam4_finetune.git
cd haiteam4_finetune
python -m pip install -r requirements.txt
python scripts/setup_model.py
python prepare_data.py
python prepare_data.py --verify-only
```

`setup_model.py`는 [고정된 공식 RT-DETR v1 R50-vd COCO 체크포인트와 소스](upstream-lock.json)를 다운로드하고 SHA256을 검사합니다. `prepare_data.py`는 COCO val2017에서 **calibration 512장**과 **평가 1,000장**을 겹치지 않게 고르고, 선택한 이미지만 추출해 파일 목록과 해시를 기록합니다. COCO 데이터 ZIP 약 1.1GB와 추가 디스크 공간이 필요합니다. 두 부분집합은 공식 COCO val 5,000장 전체 평가와 다릅니다.

```text
protocol.json                         # 모델·입력·데이터 수·평가 조건
upstream-lock.json                    # 공식 소스 커밋과 체크포인트 해시
prepare_data.py                       # COCO 다운로드·부분집합 생성·검증
scripts/setup_model.py                # 공식 모델 다운로드·검증
scripts/evaluate_predictions.py       # 공통 COCO 지표 계산
colab/COCO_Eval.ipynb                 # Colab 준비·예측 결과 평가
data/coco/val2017/                    # 선택 이미지 1,512장 (실행 후 생성)
data/coco/annotations/calibration.json
data/coco/annotations/evaluation.json
data/coco/manifest.json               # 데이터 목록·해시
weights/                              # 사전학습 가중치 (실행 후 생성)
vendor/RT-DETR/                       # 고정된 공식 소스 (실행 후 생성)
```

생성된 이미지·가중치·결과 파일은 GitHub에 올리지 않습니다. [COCO 이용 조건](https://cocodataset.org/#termsofuse)을 확인하세요.

## 팀원이 동일하게 사용할 조건

- **모델:** 공식 COCO 사전학습 RT-DETR v1 R50-vd. 재학습·클래스 교체 없이 같은 체크포인트에서 시작합니다.
- **Calibration:** `calibration.json`의 512장만 양자화 파라미터 선택에 사용합니다. `evaluation.json`의 1,000장은 calibration이나 방법 선택에 사용하지 않습니다.
- **입력:** RGB, 원본 이미지를 640×640 bilinear resize, 픽셀값 0–1. 정확한 값은 [protocol.json](protocol.json)에 고정합니다.
- **예측 출력:** 평가 이미지의 원본 크기를 기준으로 한 COCO `image_id`, 원래 COCO `category_id`, `bbox=[x,y,width,height]`, `score`. 내부 모델 출력이 xyxy라면 제출 전에 xywh로 바꿉니다.
- **정확도 지표:** COCO bbox mAP50:95, AP50, AP75, 클래스별 AP. `evaluate_predictions.py`가 모두 같은 1,000장·같은 pycocotools 규칙으로 계산합니다. AP는 모든 점수를 사용합니다.
- **추가 자원 지표:** 모델 파일 크기, 추론 호출 시간 p50/p95, FPS, 최대 RSS를 기록하려면 batch 1과 warmup 10장을 사용합니다. 서로 다른 장비·실행 엔진의 속도는 직접 비교하지 않습니다.

## 예측 결과 채점

각자 만든 모델로 **평가 1,000장**을 추론하고 COCO 형식 JSON 배열로 저장합니다. 한 탐지 결과의 예시는 다음과 같습니다.

```json
{"image_id": 397133, "category_id": 1, "bbox": [50.0, 70.0, 120.0, 90.0], "score": 0.87}
```

`image_id`는 반드시 `evaluation.json`에 있는 값을 사용하세요. JSON 전체는 위 객체들의 배열입니다. 탐지가 없는 이미지는 별도 객체 없이 생략합니다.

```powershell
python scripts/evaluate_predictions.py --predictions submissions/my_model.json --output results/my_model
```

결과 폴더에 `metrics.json`, `class_metrics.csv`, `provenance.json`이 생성됩니다. [Colab 노트북](https://colab.research.google.com/github/foxyaef/haiteam4_finetune/blob/main/colab/COCO_Eval.ipynb)에서도 같은 도구로 JSON을 채점할 수 있습니다. **노트북은 모델 자체를 실행하지 않습니다.** 팀마다 양자화 결과 형식이 다르므로, 모델 추론과 예측 JSON 생성은 각자 담당합니다.
