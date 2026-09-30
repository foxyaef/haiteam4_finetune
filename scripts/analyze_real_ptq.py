"""Compare shared COCO evaluation results with their FP32 ONNX baseline; stdlib only."""
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path


def read_bundle(source):
    source = Path(source)
    if source.is_dir():
        folders = [source] if (source / 'metrics.json').is_file() else [p.parent for p in source.glob('*/metrics.json')]
        return [{name: json.loads((folder / f'{name}.json').read_text(encoding='utf-8'))
                 for name in ('metrics', 'profile', 'provenance')} for folder in folders]
    with zipfile.ZipFile(source) as bundle:
        names = set(bundle.namelist())
        prefixes = sorted(name[:-len('metrics.json')] for name in names if name.endswith('metrics.json'))
        if not prefixes:
            raise ValueError(f'No results in {source}')
        result = []
        for prefix in prefixes:
            entry = {}
            for name in ('metrics', 'profile', 'provenance'):
                member = prefix + name + '.json'
                if member not in names or bundle.getinfo(member).file_size > 5_000_000:
                    raise ValueError(f'Missing or oversized result: {member}')
                entry[name] = json.load(io.TextIOWrapper(bundle.open(member), encoding='utf-8'))
            result.append(entry)
        return result


def analyze(inputs, output):
    results = [item for source in inputs for item in read_bundle(source)]
    baselines = [item for item in results if item['provenance']['kind'] == 'onnx_fp32']
    if len(baselines) != 1:
        raise ValueError('Supply exactly one ONNX FP32 baseline result')
    base = baselines[0]
    common = ('checkpoint_sha256', 'data_manifest_sha256', 'config_sha256', 'evaluator_sha256', 'onnxruntime')
    for item in results:
        if any(item['provenance'][key] != base['provenance'][key] for key in common):
            raise ValueError(f'Incompatible model/data/evaluator: {item["provenance"]["run_id"]}')
        if item['provenance']['providers'] != base['provenance']['providers']:
            raise ValueError('ONNX Runtime provider differs between results')
        properties = item['provenance']['model_properties']
        if properties.get('exporter_sha256') != base['provenance']['model_properties'].get('exporter_sha256'):
            raise ValueError('Quantized model used a different ONNX exporter implementation')
    base_ap = base['metrics']['mAP50_95']
    base_size = base['profile']['onnx_model_bytes']
    base_p50 = base['profile']['timing_ms_per_image']['runtime_ms']['p50']
    same_cpu = all(item['provenance'].get('cpu') == base['provenance'].get('cpu') for item in results)
    rows = []
    for item in results:
        m, p, v = item['metrics'], item['profile'], item['provenance']
        audit = v['artifact_audit']
        same_source = (item is base or v['model_properties'].get('fp32_model_sha256') == base['provenance']['model_sha256'])
        runtime_p50 = p['timing_ms_per_image']['runtime_ms']['p50']
        size = p['onnx_model_bytes']
        rows.append({'run_id': v['run_id'], 'kind': v['kind'],
                     'method': v.get('method_label', v['kind']),
                     'identity_status': v.get('identity_status', 'project_generated'),
                     'same_fp32_onnx_source': same_source,
                     'mAP50_95': m['mAP50_95'], 'delta_ap_points': round(100 * (base_ap - m['mAP50_95']), 4),
                     'AP50': m['mAP50'], 'AP75': m['mAP75'],
                     'model_mib': round(size / 2**20, 3), 'size_ratio_to_fp32': round(size / base_size, 4),
                     'runtime_p50_ms': round(runtime_p50, 3),
                     'runtime_p95_ms': round(p['timing_ms_per_image']['runtime_ms']['p95'], 3),
                     'runtime_speedup_vs_fp32': round(base_p50 / runtime_p50, 4),
                     'observed_fps': round(p['observed_fps_including_python'], 3),
                     'peak_rss_mib': round(p['peak_process_rss_bytes'] / 2**20, 3),
                     'runtime_integer_nodes': sum(v['optimized_runtime_op_counts'].get(k, 0) for k in
                                                  ('QLinearConv', 'QLinearMatMul', 'MatMulInteger', 'ConvInteger',
                                                   'QGemm', 'MatMulIntegerToFloat', 'DynamicQuantizeMatMul')),
                     'int4_matmul_nodes': audit.get('matmul_nbits_nodes', 0),
                     'int4_weight_elements': audit.get('int4_weight_elements', 0)})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# RT-DETR COCO 공통 평가', '',
             'COCO val2017 평가 1,000장과 ONNX Runtime CPU 실행을 기준으로 비교합니다.',
             'ΔAP point = (FP32 AP − 실험 AP) × 100. 양수는 정확도 손실입니다.',
             '팀이 등록한 모델의 양자화 기법·원본 모델 정보는 자기 신고입니다. 그래프 연산 수와 원본 파일을 별도로 확인하세요.', '',
             '| 모델·방법 | mAP50:95 | ΔAP point | 크기 MiB | FP32 대비 크기 | 런타임 p50 ms | FPS | peak RSS MiB | INT4 MatMul 수 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        label = row['method'].replace('|', '/')
        lines.append(f'| {label} | {row["mAP50_95"]:.4f} | {row["delta_ap_points"]:.2f} | {row["model_mib"]:.1f} | {row["size_ratio_to_fp32"]:.2f}× | {row["runtime_p50_ms"]:.1f} | {row["observed_fps"]:.1f} | {row["peak_rss_mib"]:.1f} | {row["int4_matmul_nodes"]} |')
    if any(not row['same_fp32_onnx_source'] for row in rows):
        lines.extend(['', '주의: 업로드 모델의 원본 FP32 ONNX 해시가 이번 기준 파일과 다릅니다. 공식 체크포인트·평가 설정·export 코드가 같아 비교를 허용했지만, 엄밀한 동일 그래프 비교는 같은 FP32 ONNX에서 다시 양자화해야 합니다.'])
    if any(row['identity_status'] == 'team_declared' for row in rows):
        lines.extend(['', '주의: 팀 등록 모델의 원본·양자화 방법은 등록자가 선언한 정보이며 평가기가 증명하지 않습니다. 발표 전에 모델 생성 기록과 그래프 적용 범위를 확인하세요.'])
    if not same_cpu:
        lines.extend(['', '주의: 결과의 CPU 모델이 서로 달라 속도·메모리 수치를 직접 비교하기 어렵습니다. 정확도 지표는 같은 평가 데이터 기준으로 볼 수 있습니다.'])
    (output / 'analysis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(output / 'summary.csv')
    print(output / 'analysis.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='+')
    parser.add_argument('--output', default='runs/real_ptq_analysis')
    args = parser.parse_args()
    analyze(args.results, args.output)


if __name__ == '__main__':
    main()
