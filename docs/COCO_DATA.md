# COCO 2017 데이터 준비

`prepare_data.py`는 이번 COCO 사전학습 모델 실험에 필요한 **데이터 다운로드·검증·폴더 구성·부분집합 고정**을 한 번에 수행합니다.

```text
python prepare_data.py
python prepare_data.py --verify-only
```

첫 명령은 COCO 공식 S3에 있는 `annotations_trainval2017.zip`과 `val2017.zip`을 내려받습니다. 공개된 아카이브의 [크기와 SHA256](https://huggingface.co/datasets/pcuenq/coco-2017-mirror/commit/5200d2cffae8121713bec767b4693bdafc0eeb0b)을 코드에 고정했고, 내려받은 파일을 검사합니다. HTTPS 인증 검사를 끄지 않습니다. 다운로드가 끊기면 `.part` 파일에서 이어받습니다. ZIP 안의 `annotations/instances_val2017.json`을 읽고, seed 42로 val 5,000장 중 **calibration 512장**과 **평가 1,000장**을 겹치지 않게 고릅니다. 전체 5,000장을 풀지 않고 선택한 JPEG 1,512장만 `data/coco/val2017/`에 추출하며 ZIP CRC를 검사합니다.

```text
data/coco/
  downloads/annotations_trainval2017.zip
  downloads/val2017.zip
  val2017/*.jpg
  annotations/calibration.json
  annotations/evaluation.json
  manifest.json
```

`manifest.json`에는 원본 아카이브, 원본 val 어노테이션, 각 선택 목록과 COCO JSON의 SHA256이 기록됩니다. 동일 파일이 이미 있으면 그대로 검사하고 내용이 다른 파일을 덮어쓰지 않습니다. 팀원들이 같은 목록과 해시를 사용하는지 비교하세요. 전체 이미지 아카이브와 선택 JPEG를 함께 보관하므로 대략 2GB 이상의 여유 공간을 확보하세요.

Calibration은 **COCO val 부분집합**이지만 평가 이미지와 분리되어 있습니다. 이는 COCO train2017 전체 18GB를 받지 않기 위한 연구용 선택이며 공식 COCO val 전체 mAP와 같은 프로토콜이 아닙니다. train에서 calibration을 뽑는 정식 평가로 바꾸려면 모든 팀원이 데이터 구성과 결과를 새로 만들어야 합니다. [COCO 2017 데이터 안내](https://cocodataset.org/dataset/detection-2017.htm), [원본 공식 모델의 COCO 데이터 구조](https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/README.md)를 참고하세요.
