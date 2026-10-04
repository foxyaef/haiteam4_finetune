"""Evaluate uploaded models in separate processes and preserve partial results."""
import argparse
import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import ROOT, sha256


METRIC_FIELDS = ['mAP50_95', 'AP50', 'AP75', 'peak_RSS_MiB', 'RSS_increase_MiB',
                 'GPU_process_peak_MiB', 'latency_p50_ms', 'latency_p95_ms', 'FPS',
                 'end_to_end_p50_ms', 'end_to_end_FPS', 'CPU_kernel_events', 'CUDA_kernel_events',
                 'cuda_uses_cpu_nodes', 'placement_status', 'has_int8_evidence', 'has_int4_evidence']


def run_model(command, log_path):
    env = dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    with log_path.open('w', encoding='utf-8') as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', env=env) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            return process.wait()


def evaluate_suite(models, output_dir, provider='cpu', data_dir=None, max_images=None, repeats=1, threads=4):
    models = sorted([Path(model).resolve() for model in models], key=lambda path: (path.stem != 'fp32', path.name))
    if not models or len({path.stem for path in models}) != len(models):
        raise ValueError('Select at least one ONNX file with unique model names')
    if any(not path.is_file() or path.suffix.lower() != '.onnx' for path in models):
        raise ValueError('All selected models must be existing ONNX files')
    if provider not in ('cpu', 'cuda') or repeats < 1 or threads < 1:
        raise ValueError('Invalid provider/repeats/threads')
    output = Path(output_dir).resolve()
    if output.exists() or output.with_suffix('.zip').exists():
        raise FileExistsError('Choose a new run directory: ' + str(output))
    output.mkdir(parents=True)
    (output / 'logs').mkdir()
    summary, failures = [], []
    for model in models:
        destination = output / model.stem
        log_path = output / 'logs' / (model.stem + '.log')
        print(f'\nEvaluating: {model.name}', flush=True)
        command = [sys.executable, str(ROOT / 'scripts/evaluate_model.py'), '--model', str(model),
                   '--output', str(destination), '--provider', provider, '--repeats', str(repeats),
                   '--threads', str(threads)]
        if data_dir is not None:
            command += ['--data-dir', str(data_dir)]
        if max_images is not None:
            command += ['--max-images', str(max_images)]
        row = {'model': model.stem, 'status': 'failed', 'provider': provider,
               'image_count': None, 'full_evaluation': None, 'latency_repeats': repeats,
               'model_sha256': sha256(model), 'file_MiB': model.stat().st_size / 1048576,
               **{field: None for field in METRIC_FIELDS}, 'error': ''}
        try:
            code = run_model(command, log_path)
            if code:
                tail = '\n'.join(log_path.read_text(encoding='utf-8').splitlines()[-12:])
                raise RuntimeError(f'Evaluator exited with code {code}.\n{tail}')
            metrics = json.loads((destination / 'metrics.json').read_text(encoding='utf-8'))
            resource = metrics['resource']
            placement = json.loads((destination / 'provider_placement.json').read_text(encoding='utf-8'))
            audit = json.loads((destination / 'quantization_audit.json').read_text(encoding='utf-8'))
            kernel_counts = placement['kernel_event_counts']
            recorded = placement['status'] == 'recorded'
            row.update(status='passed', image_count=metrics['image_count'], full_evaluation=metrics['full_evaluation'],
                       mAP50_95=metrics['mAP50_95'], AP50=metrics['mAP50'], AP75=metrics['mAP75'],
                       peak_RSS_MiB=resource['rss_peak_bytes'] / 1048576,
                       RSS_increase_MiB=resource['rss_increase_bytes'] / 1048576,
                       GPU_process_peak_MiB=(resource['gpu_process_peak_bytes'] / 1048576
                                            if resource['gpu_process_peak_bytes'] is not None else None),
                       latency_p50_ms=resource['inference_ms_p50'], latency_p95_ms=resource['inference_ms_p95'],
                       FPS=resource['inference_fps'], end_to_end_p50_ms=resource['end_to_end_ms_p50'],
                       end_to_end_FPS=resource['end_to_end_fps'],
                       CPU_kernel_events=kernel_counts.get('CPUExecutionProvider', 0) if recorded else None,
                       CUDA_kernel_events=kernel_counts.get('CUDAExecutionProvider', 0) if recorded else None,
                       cuda_uses_cpu_nodes=placement['cuda_uses_cpu_nodes'], placement_status=placement['status'],
                       has_int8_evidence=audit['has_int8_evidence'], has_int4_evidence=audit['has_int4_evidence'])
        except Exception as error:
            row['error'] = str(error)
            failures.append({'model': model.name, 'error': str(error), 'log': str(log_path.relative_to(output))})
            print(f'FAILED: {model.name}. Continuing with remaining models; see {log_path.name}', flush=True)
        summary.append(row)
        write_summary(output, summary)
    baseline = next((row for row in summary if row['model'] == 'fp32' and row['status'] == 'passed'
                     and not row['has_int8_evidence'] and not row['has_int4_evidence']), None)
    for row in summary:
        comparable = baseline is not None and row['status'] == 'passed'
        row['delta_mAP50_95'] = row['mAP50_95'] - baseline['mAP50_95'] if comparable else None
        row['delta_mAP_pp'] = row['delta_mAP50_95'] * 100 if comparable else None
        row['size_vs_fp32'] = row['file_MiB'] / baseline['file_MiB'] if baseline else None
    write_summary(output, summary)
    (output / 'errors.json').write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = {'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'provider': provider, 'requested_max_images': max_images, 'latency_repeats': repeats,
                'threads': threads, 'models_passed': sum(row['status'] == 'passed' for row in summary),
                'models_failed': len(failures), 'baseline_available': baseline is not None,
                'protocol_sha256': sha256(ROOT / 'protocol.json'),
                'note': 'mAP is in [0,1]; delta_mAP_pp uses percentage points. Partial runs are execution checks. CUDA may include CPU nodes; inspect provider_placement.json. Sampled GPU memory may be unavailable.'}
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    shutil.copyfile(ROOT / 'protocol.json', output / 'protocol.json')
    archive = shutil.make_archive(str(output), 'zip', root_dir=output)
    print(f'\nSaved {archive}; passed={manifest["models_passed"]}, failed={len(failures)}', flush=True)
    return summary, archive


def write_summary(output, summary):
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--provider', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--max-images', type=int)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    summary, _ = evaluate_suite(args.models, args.output, args.provider, args.data_dir,
                                args.max_images, args.repeats, args.threads)
    if any(row['status'] == 'failed' for row in summary):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
