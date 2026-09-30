"""Compare downloaded Colab result ZIPs or result directories locally (standard library only)."""
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path


def read_results(source):
    source = Path(source)
    if source.is_dir():
        folders = [source] if (source / 'metrics.json').is_file() else [p.parent for p in source.glob('*/metrics.json')]
        if not folders:
            raise ValueError(f'No result folders in {source}')
        return [{name: json.loads((folder / f'{name}.json').read_text(encoding='utf-8'))
                 for name in ('metrics', 'profile', 'provenance')} for folder in folders]
    with zipfile.ZipFile(source) as zipped:
        prefixes = {p[:-len('metrics.json')] for p in zipped.namelist() if p.endswith('metrics.json')}
        if not prefixes:
            raise ValueError(f'{source}: no metrics.json')
        results = []
        for prefix in sorted(prefixes):
            found = {}
            for name in ('metrics', 'profile', 'provenance'):
                member = prefix + name + '.json'
                if member not in zipped.namelist() or zipped.getinfo(member).file_size > 5_000_000:
                    raise ValueError(f'{source}: missing or oversized {member}')
                found[name] = json.load(io.TextIOWrapper(zipped.open(member), encoding='utf-8'))
            results.append(found)
        return results


def analyze(sources, output):
    results = [result for item in sources for result in read_results(item)]
    baselines = [item for item in results if item['provenance']['kind'] == 'official_fp32']
    if len(baselines) != 1:
        raise ValueError('Provide exactly one FP32 baseline result')
    baseline = baselines[0]
    identity = ('checkpoint_sha256', 'data_manifest_sha256', 'config_sha256', 'evaluator_sha256')
    for item in results:
        if any(item['provenance'][key] != baseline['provenance'][key] for key in identity):
            raise ValueError(f'Model/data/config mismatch: {item["provenance"]["run_id"]}')
    hardware = ('device', 'gpu', 'torch', 'torchvision')
    comparable = all(all(item['provenance'][key] == baseline['provenance'][key] for key in hardware) for item in results)
    rows = []
    for item in results:
        p, m, r = item['provenance'], item['metrics'], item['profile']
        rows.append({'run_id': p['run_id'], 'kind': p['kind'], 'group': (p['coverage'] or {}).get('group', ''),
                     'bits': (p['coverage'] or {}).get('bits', ''), 'mAP50_95': m['mAP50_95'],
                     'delta_ap_points': round(100 * (baseline['metrics']['mAP50_95'] - m['mAP50_95']), 4),
                     'AP50': m['mAP50'], 'AP75': m['mAP75'],
                     'model_p50_ms': r['timing_ms_per_image']['model_ms']['p50'],
                     'total_p50_ms': r['timing_ms_per_image']['total_ms']['p50'],
                     'total_p95_ms': r['timing_ms_per_image']['total_ms']['p95'],
                     'observed_fps': r['observed_fps_including_python'],
                     'peak_rss_mib': r['peak_process_rss_bytes'] / 2**20,
                     'cuda_peak_allocated_mib': r['cuda_peak_allocated_bytes'] / 2**20 if r['cuda_peak_allocated_bytes'] is not None else '',
                     'artifact_mib': r['artifact_bytes'] / 2**20})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Colab RT-DETR 평가 비교', '',
             f'- 기준: {baseline["provenance"]["run_id"]}, COCO val2017 고정 1,000장',
             f'- 런타임 비교 조건: {"동일 장치/프레임워크" if comparable else "서로 다름 — 속도·메모리 직접 비교 주의"}',
             '- ΔAP point = (FP32 mAP50:95 − 실험 mAP50:95) × 100. 양수는 정확도 손실.',
             '- fake PTQ는 FP32 커널로 실행됩니다. 시간과 파일 크기는 실제 INT4/INT6 배포 성능이 아닙니다.',
             '- 최고 RSS는 20ms 간격 샘플값이며 매우 짧은 순간의 피크는 놓칠 수 있습니다.', '',
             '| 실행 | mAP50:95 | ΔAP point | 전체 p50 (ms) | 전체 p95 (ms) | FPS | peak RSS (MiB) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["run_id"]} | {row["mAP50_95"]:.4f} | {row["delta_ap_points"]:.2f} | {row["total_p50_ms"]:.1f} | {row["total_p95_ms"]:.1f} | {row["observed_fps"]:.2f} | {row["peak_rss_mib"]:.1f} |')
    (output / 'analysis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(output / 'summary.csv')
    print(output / 'analysis.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='+', help='FP32 baseline plus one or more artifact ZIPs or directories')
    parser.add_argument('--output', default='runs/colab_analysis')
    args = parser.parse_args()
    analyze(args.results, args.output)


if __name__ == '__main__':
    main()
