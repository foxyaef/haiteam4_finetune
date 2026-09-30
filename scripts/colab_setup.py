"""Download the pinned official RT-DETR source and COCO checkpoint on Linux/Colab."""
import json
import subprocess
import urllib.request

from common import ROOT, sha256


def run(*args):
    subprocess.run(args, check=True)


def main():
    lock = json.loads((ROOT / 'upstream-lock.json').read_text(encoding='utf-8'))
    vendor = ROOT / 'vendor/RT-DETR'
    if not vendor.exists():
        vendor.parent.mkdir(parents=True, exist_ok=True)
        run('git', 'clone', lock['repository'], str(vendor))
        run('git', '-C', str(vendor), 'checkout', '--detach', lock['commit'])
    actual = subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(vendor), 'status', '--porcelain'], text=True)
    if actual != lock['commit'] or dirty.strip():
        raise ValueError('Existing vendor/RT-DETR is not the clean pinned source')
    target = ROOT / 'weights' / lock['checkpoint']
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or sha256(target) != lock['checkpoint_sha256']:
        temp = target.with_suffix('.download')
        try:
            with urllib.request.urlopen(lock['checkpoint_url'], timeout=120) as source, temp.open('wb') as dest:
                while chunk := source.read(1024 * 1024):
                    dest.write(chunk)
            if sha256(temp) != lock['checkpoint_sha256']:
                raise ValueError('Downloaded checkpoint SHA256 mismatch')
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
    print(f'Pinned source and checkpoint ready: {target}')


if __name__ == '__main__':
    main()
