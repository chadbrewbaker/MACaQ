"""Create a saved dense MACaQ checkpoint with W4A4 or weight-only inference.

Reuses the exact packed weights; does not recalibrate or requantize. Large
safetensors files are hardlinked when possible, copied across filesystems.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def select(model, output, precision):
    if precision not in ('w4a4', 'w4a16'):
        raise ValueError('precision must be w4a4 or w4a16')
    config = json.loads((model / 'config.json').read_text())
    qc = config.get('quantization_config', {})
    if qc.get('quant_method') != 'mxfp4' or qc.get('macaq_dense_w4a4') is not True:
        raise ValueError('source must be a saved dense MACaQ MXFP4 checkpoint')
    if not list(model.glob('*.safetensors')):
        raise ValueError('source has no safetensors weights')
    if output.exists() or output.resolve().is_relative_to(model.resolve()):
        raise ValueError('choose a fresh output directory outside the source')
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=output.name + '.pending-', dir=output.parent))
    try:
        weights = []
        for source in sorted(model.iterdir()):
            if not source.is_file() or source.name in ('config.json', 'macaq_precision.json'):
                continue
            target = pending / source.name
            storage = 'copy'
            if source.suffix == '.safetensors':
                try:
                    os.link(source.resolve(), target)
                    storage = 'hardlink'
                except OSError:
                    shutil.copy2(source, target)
                weights.append(dict(name=source.name, sha256=sha256(source),
                                    bytes=source.stat().st_size, storage=storage))
            else:
                shutil.copy2(source, target)
        qc['macaq_activation_precision'] = precision
        config['quantization_config'] = qc
        (pending / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        (pending / 'macaq_precision.json').write_text(json.dumps(dict(
            completed=True, activation_precision=precision, source=str(model.resolve()),
            source_config_sha256=sha256(model / 'config.json'),
            calibration='reused source recipe and packed weights', weights=weights), indent=2) + '\n')
        pending.rename(output)
    except BaseException:
        shutil.rmtree(pending)
        raise
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--precision', choices=('w4a4', 'w4a16'), required=True)
    args = ap.parse_args()
    print(select(args.model, args.output, args.precision))


if __name__ == '__main__':
    main()
