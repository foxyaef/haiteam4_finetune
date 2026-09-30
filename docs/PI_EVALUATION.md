# Raspberry Pi 측정 계획

현재 W4A4/W6A6/W8A8은 정확도를 확인하는 fake quantization입니다. **Pi의 속도·전력·모델 크기 이득은 별도의 실제 배포 모델과 런타임이 준비된 뒤 측정**해야 합니다. FP32와 양자화 모델에서 입력 640×640, 같은 COCO 평가 이미지, 전처리·후처리, 장치·스레드 수를 고정합니다.

| 우선 지표 | 측정·기록 방법 |
|---|---|
| 지연시간 | batch 1의 p50·p95와 평균. 이미지 로드/resize, 모델, 후처리, 전체 시간을 구분 |
| 지속 FPS | 최소 5~10분 연속 처리한 이미지 수/경과 시간. 짧은 burst와 구분 |
| 메모리·실행 가능성 | 적재 전후/추론 중 peak RSS, 시스템 사용량, 스왑, OOM. **RAM 2GB**에서 특히 중요 |
| 온도·클록·throttling | 추론 중 시간별 SoC 온도, ARM 클록, 제한 플래그와 발생 시점 |
| 전력·에너지 | 외부 전력계의 유휴/추론 W, 처리 이미지당 Wh. 내부 센서 값만으로 소비전력을 대체하지 않음 |

실제 파일 크기, 모델 적재 시간(cold start), 정확도 mAP50:95·클래스별 AP도 함께 기록합니다. 하드웨어 모델, 64비트 OS, 냉각, 전원 공급장치, 저장매체, 런타임·버전, CPU 스레드 수를 표에 고정하고 조건별로 최소 3회 반복하세요. Pi에서 스왑이 발생하면 지연시간이 크게 달라질 수 있으므로 FPS와 함께 보여줘야 합니다.

Raspberry Pi OS의 `vcgencmd measure_temp`, `vcgencmd measure_clock arm`, `vcgencmd get_throttled`는 온도·클록·전압/열 제한 상태를 확인할 때 쓸 수 있습니다. `get_throttled` 값에는 **현재와 과거에 발생한 상태가 서로 다른 비트**로 기록됩니다. [Raspberry Pi 공식 문서](https://www.raspberrypi.com/documentation/computers/os.html#vcgencmd)
