"""Aggregate completed phase-one runs without silently filling missing results."""
import csv
import json
from pathlib import Path

from common import ROOT, dump


def main():
    root = ROOT / 'runs/phase1'
    baseline_path = root / 'fp32/metrics.json'
    if not baseline_path.is_file():
        raise FileNotFoundError('Run the shared FP32 phase-one baseline first')
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    rows = []
    for group in ('fp32','all','backbone','encoder','decoder','head'):
        for bits in ((None,) if group == 'fp32' else (8,6,4)):
            name = 'fp32' if bits is None else f'{group}_w{bits}a{bits}'
            filename = root / name / 'metrics.json'
            if not filename.is_file():
                rows.append(dict(run_id=name, status='missing', mAP50_95='', delta_ap_points='', AP50='', AP75=''))
                continue
            result = json.loads(filename.read_text(encoding='utf-8'))
            rows.append(dict(run_id=name, status='complete',
                             mAP50_95=result['mAP50_95'],
                             delta_ap_points=100*(baseline['mAP50_95']-result['mAP50_95']),
                             AP50=result['mAP50'], AP75=result['mAP75']))
    out = root / 'summary.csv'
    with out.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(out)


if __name__ == '__main__':
    main()
