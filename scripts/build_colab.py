"""Build a portable Colab notebook containing the exact evaluation source snapshot."""
import ast
import base64
import hashlib
import io
import json
import textwrap
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = ['protocol.json', 'upstream-lock.json', 'prepare_data.py', 'requirements.txt',
         'requirements-eval-cpu.txt', 'requirements-eval-gpu.txt', 'scripts/common.py',
         'scripts/metrics.py', 'scripts/model_audit.py', 'scripts/evaluate_model.py', 'scripts/evaluate_suite.py']


SETUP = r'''#@title 2. 환경과 COCO 이미지 준비
import base64, gc, hashlib, io, json, os, pathlib, shutil, subprocess, sys, zipfile
if provider not in ('cpu', 'cuda') or evaluation_mode not in ('smoke', 'full'):
    raise ValueError('1번 셀에서 provider와 evaluation_mode를 선택하세요.')
if not isinstance(latency_repeats, int) or not 1 <= latency_repeats <= 5:
    raise ValueError('latency_repeats는 1~5 정수로 설정하세요.')
if not isinstance(cpu_threads, int) or not 1 <= cpu_threads <= 32:
    raise ValueError('cpu_threads는 1~32 정수로 설정하세요.')
if not (3, 10) <= sys.version_info[:2] <= (3, 13):
    raise RuntimeError('이 환경은 Python 3.10~3.13을 사용합니다. Colab 런타임의 Python 버전을 확인하세요.')
if provider == 'cuda':
    if not shutil.which('nvidia-smi'):
        raise RuntimeError('Colab 런타임 유형을 GPU로 바꾸거나 1번 셀에서 provider="cpu"를 선택하세요.')
    subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], check=True)

# 평가 코드 스냅샷: GitHub 최신 버전이나 기존 clone 상태에 영향을 받지 않습니다.
payload_sha256 = '__HASH__'
payload_encoded = (
__PAYLOAD__
)
payload = base64.b64decode(payload_encoded)
if hashlib.sha256(payload).hexdigest() != payload_sha256:
    raise RuntimeError('노트북 평가 코드의 SHA256 검증에 실패했습니다. 원본 노트북을 다시 받으세요.')
work = pathlib.Path(os.environ.get('RTDETR_EVAL_WORK_DIR', '/content/rtdetr_evaluation')).resolve()
runtime = work / ('runtime_' + payload_sha256[:12])
runtime.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
    for member in bundle.infolist():
        target = (runtime / member.filename).resolve()
        if not target.is_relative_to(runtime):
            raise ValueError('평가 코드에 잘못된 파일 경로가 있습니다.')
        contents = bundle.read(member)
        if target.exists() and target.read_bytes() != contents:
            raise RuntimeError('기존 평가 코드가 수정되어 있습니다: ' + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
del payload, payload_encoded
gc.collect()

requirements = runtime / ('requirements-eval-gpu.txt' if provider == 'cuda' else 'requirements-eval-cpu.txt')
dependency_hash = hashlib.sha256(requirements.read_bytes() + (runtime / 'requirements.txt').read_bytes()).hexdigest()[:12]
environment = work / ('env_' + provider + '_py' + str(sys.version_info.minor) + '_' + dependency_hash)
eval_python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
if not eval_python.exists():
    create = subprocess.run([sys.executable, '-m', 'venv', str(environment)], capture_output=True, text=True)
    if create.returncode:
        # Some Colab Python installations omit ensurepip; virtualenv can bootstrap it.
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'virtualenv'], check=True)
        subprocess.run([sys.executable, '-m', 'virtualenv', str(environment)], check=True)
subprocess.run([str(eval_python), '-m', 'pip', 'install', '--quiet', '-r', str(requirements)], check=True)
subprocess.run([str(eval_python), '-m', 'pip', 'check'], check=True)
packages_file = work / ('environment_' + provider + '.txt')
packages_file.write_text(subprocess.check_output([str(eval_python), '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
data_dir = pathlib.Path(os.environ.get('RTDETR_EVAL_DATA_DIR', str(work / 'data/coco'))).resolve()
subprocess.run([str(eval_python), str(runtime / 'prepare_data.py'), '--data-dir', str(data_dir)], check=True)
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
print('준비 완료: 공통 calibration 512장 / 평가 1,000장. 다음 셀에서 모델 파일을 선택하세요.')
'''


UPLOAD = r'''#@title 3. 모델 파일 선택
model_source = 'upload' #@param ["upload", "drive"]
drive_model_folder = '/content/drive/MyDrive/rtdetr_models' #@param {type:"string"}
import datetime, gc, os, pathlib
from google.colab import files, drive
if model_source == 'drive':
    drive.mount('/content/drive')
    model_dir = pathlib.Path(drive_model_folder)
    if not model_dir.is_dir():
        raise FileNotFoundError('모델 폴더가 없습니다: ' + str(model_dir))
    model_paths = [path.resolve() for path in model_dir.iterdir() if path.is_file() and path.suffix.lower() == '.onnx']
elif model_source == 'upload':
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    model_dir = work / 'uploads' / stamp
    model_dir.mkdir(parents=True)
    previous_directory = pathlib.Path.cwd()
    try:
        os.chdir(model_dir)
        uploaded = files.upload()  # fp32.onnx + 양자화 모델 6개를 함께 선택
    finally:
        os.chdir(previous_directory)
    model_paths = [model_dir / name for name in uploaded if pathlib.Path(name).name == name
                   and pathlib.Path(name).suffix.lower() == '.onnx']
    del uploaded
    gc.collect()  # 대용량 업로드의 임시 바이트 버퍼 해제
else:
    raise ValueError('model_source는 upload 또는 drive를 선택하세요.')
model_paths = sorted(model_paths, key=lambda path: (path.stem != 'fp32', path.name))
if not model_paths:
    raise ValueError('단일 파일에 가중치가 포함된 .onnx 모델을 선택하세요. .pth 파일은 직접 평가하지 않습니다.')
if len({path.stem for path in model_paths}) != len(model_paths):
    raise ValueError('모델 파일 이름이 중복됩니다.')
if not any(path.name == 'fp32.onnx' for path in model_paths):
    print('fp32.onnx가 없어 ΔmAP와 FP32 대비 크기 비율은 계산하지 않습니다.')
print('평가할 모델:')
for path in model_paths:
    print(' -', path.name, f'({path.stat().st_size / 1048576:.2f} MiB)')
'''


EVALUATE = r'''#@title 4. 평가·결과 다운로드
import datetime, html, json, pathlib, shutil, subprocess
from IPython.display import display, HTML, Image as DisplayImage
if provider not in ('cpu', 'cuda') or evaluation_mode not in ('smoke', 'full'):
    raise ValueError('평가 설정을 확인하세요.')
settings = {'provider': provider, 'evaluation_mode': evaluation_mode,
            'latency_repeats': latency_repeats, 'cpu_threads': cpu_threads}
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
run_dir = work / 'results' / (evaluation_mode + '_' + provider + '_' + stamp)
command = [str(eval_python), str(runtime / 'scripts/evaluate_suite.py'), '--models', *map(str, model_paths),
           '--output', str(run_dir), '--provider', provider, '--data-dir', str(data_dir),
           '--repeats', str(latency_repeats), '--threads', str(cpu_threads)]
if evaluation_mode == 'smoke':
    command += ['--max-images', '3']
    print('3장 실행 확인입니다. 이 결과의 mAP·속도는 최종 연구 결과로 사용하지 마세요.')
else:
    print('공통 평가 이미지 1,000장으로 평가합니다. 모델별로 독립 프로세스를 사용합니다.')
completed = subprocess.run(command, check=False)
run_dir.mkdir(parents=True, exist_ok=True)
shutil.copyfile(packages_file, run_dir / 'environment_packages.txt')
shutil.copyfile(runtime / 'embedded_manifest.json', run_dir / 'embedded_manifest.json')
(run_dir / 'notebook_run.json').write_text(json.dumps({**settings, 'exit_code': completed.returncode,
    'evaluation_bundle_sha256': payload_sha256}, indent=2), encoding='utf-8')
summary_path = run_dir / 'summary.json'
summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else []
columns = [('model', '모델'), ('status', '상태'), ('mAP50_95', 'mAP50:95'), ('delta_mAP_pp', 'ΔmAP(pp)'),
           ('file_MiB', '파일 MiB'), ('peak_RSS_MiB', 'RAM MiB'), ('GPU_process_peak_MiB', 'GPU MiB'),
           ('latency_p50_ms', 'p50 ms'), ('latency_p95_ms', 'p95 ms'), ('FPS', 'FPS'),
           ('cuda_uses_cpu_nodes', 'CUDA 중 CPU 사용')]
def render(value):
    return '—' if value is None else html.escape(f'{value:.4f}' if isinstance(value, float) else str(value))
table = '<div style="overflow:auto"><table><thead><tr>' + ''.join('<th>' + label + '</th>' for _, label in columns) + '</tr></thead><tbody>'
for row in summary:
    table += '<tr>' + ''.join('<td>' + render(row.get(key)) + '</td>' for key, _ in columns) + '</tr>'
display(HTML(table + '</tbody></table></div>'))
for row in summary:
    if row['status'] != 'passed':
        print('실패:', row['model'], '— 결과 ZIP의 errors.json과 logs/를 확인하세요.')
if completed.returncode:
    print('일부 평가가 실패했습니다. 성공 결과와 오류 로그를 모두 내려받습니다.')
for path in model_paths:
    preview = run_dir / path.stem / 'preview_1.jpg'
    if preview.exists():
        print(path.name, '탐지 미리보기')
        display(DisplayImage(filename=str(preview)))
archive = shutil.make_archive(str(run_dir), 'zip', root_dir=run_dir)
print('결과 ZIP:', archive)
files.download(archive)
'''


def cell(kind, text, index, form=False):
    result = {'cell_type': kind, 'id': hashlib.sha256(f'{index}:{text}'.encode()).hexdigest()[:12],
              'metadata': {'cellView': 'form'} if form else {}, 'source': text.splitlines(keepends=True)}
    if kind == 'code':
        ast.parse(text)
        result.update(execution_count=None, outputs=[])
    return result


def build():
    contents = {name: (ROOT / name).read_bytes().replace(b'\r\n', b'\n') for name in FILES}
    manifest = {'files_sha256': {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()},
                'source': 'foxyaef/haiteam4_finetune', 'format': 'self-contained evaluation source snapshot',
                'note': 'No model weights or dataset images embedded. Same locked COCO selection and deploy interface.'}
    contents['embedded_manifest.json'] = json.dumps(manifest, indent=2).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, value in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 4, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, value)
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    encoded = '\n'.join('    ' + repr(part) for part in textwrap.wrap(base64.b64encode(data).decode(), 100))
    setup = SETUP.replace('__HASH__', digest).replace('__PAYLOAD__', encoded)
    entries = [
        ('markdown', '# RT-DETR 모델 업로드 평가\n\nFP32와 Hybrid Encoder의 **Transformer만 / CNN만 / 둘 다**, 각각 W8A8·W4A16으로 만든 ONNX 파일을 동일한 COCO val2017 평가 이미지에서 비교합니다.\n\n**사용 순서:** 설정 → 환경·이미지 준비 → 모델 선택 → 평가·ZIP 다운로드. 기본은 `full` 1,000장입니다. 먼저 `smoke` 3장으로 실행을 확인할 수도 있습니다. 평가 코드가 이 노트북에 포함되어 있어 파일을 Colab에 직접 업로드해 사용할 수 있습니다.\n\nGPU 사용 시 **런타임 → 런타임 유형 변경 → GPU**를 선택하세요. CPU 비교는 `provider=cpu`로 통일합니다.\n', False),
        ('code', '#@title 1. 평가 설정\nprovider = \'cuda\' #@param ["cuda", "cpu"]\nevaluation_mode = \'full\' #@param ["full", "smoke"]\nlatency_repeats = 1 #@param {type:"integer"}\ncpu_threads = 4 #@param {type:"integer"}\n', True),
        ('markdown', '## 환경과 데이터\n\n평가 전용 Python 환경을 만듭니다. 최초 COCO 다운로드는 약 1.1GB이며, 기존 프로젝트와 같은 calibration 512장·평가 1,000장을 추출합니다. `smoke`도 이 평가 목록에서 앞 3장을 사용합니다.\n', False),
        ('code', setup, True),
        ('markdown', '## 모델 선택\n\n`artifacts/cal512/`의 **fp32.onnx와 양자화 ONNX 6개**를 함께 선택하세요. `upload`는 PC 파일 업로드, `drive`는 Google Drive 폴더입니다. 대용량 파일은 Drive가 편리합니다.\n\n| 범위 | W8A8 | W4A16 |\n|---|---|---|\n| Transformer | encoder_transformer_w8a8.onnx | encoder_transformer_w4a16.onnx |\n| CNN | encoder_cnn_w8a8.onnx | encoder_cnn_w4a16.onnx |\n| 둘 다 | encoder_both_w8a8.onnx | encoder_both_w4a16.onnx |\n\n각 파일은 가중치가 포함된 단일 ONNX여야 합니다. 입력은 `images`와 원본 `[width,height]`의 `orig_target_sizes`, 출력은 `labels/boxes/scores`입니다. `.pth`는 직접 평가하지 않습니다.\n', False),
        ('code', UPLOAD, True),
        ('markdown', '## 평가 결과\n\nmAP50:95·AP50·AP75·클래스별 AP, 실제 파일 크기, RAM(RSS), 가능한 GPU 메모리, p50/p95 지연·FPS를 기록합니다. 지연은 별도 워밍업 후 측정하며 `latency_repeats`는 시간 측정 반복 횟수입니다. mAP에는 이미지당 한 번의 예측만 사용합니다.\n\n모델 하나에서 오류가 나도 나머지 모델을 계속 평가하고 성공 결과와 오류 로그를 ZIP으로 저장합니다. CUDA에서 CPU로 실행된 연산은 별도 profiling 결과에 기록합니다. **W4A16의 INT4 저장이 전용 INT4 커널 사용을 보장하지 않으므로 실행 결과로 비교하세요.**\n\n표의 AP는 0~1, ΔmAP는 percentage point(pp), 크기는 MiB입니다. GPU 메모리를 측정하지 못한 값은 빈칸이며 0으로 간주하지 않습니다. CPU와 CUDA 결과는 같은 표에서 성능 우열을 비교하지 마세요.\n', False),
        ('code', EVALUATE, True),
    ]
    notebook = {'cells': [cell(kind, text, i, form) for i, (kind, text, form) in enumerate(entries)],
                'metadata': {'colab': {'name': 'COCO_Eval.ipynb', 'toc_visible': True, 'provenance': []},
                             'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python'}, 'accelerator': 'GPU',
                             'evaluation_bundle_sha256': digest}, 'nbformat': 4, 'nbformat_minor': 5}
    target = ROOT / 'colab/COCO_Eval.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'Built {target.name}: {target.stat().st_size} bytes; embedded snapshot {digest}')


if __name__ == '__main__':
    build()
