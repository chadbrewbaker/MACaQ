"""Fingerprint source checkpoints and save model provenance."""
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def source_fingerprint(directory):
    directory=Path(directory)
    config=json.loads((directory/'config.json').read_text())
    if config.get('quantization_config'):
        raise ValueError('the source control must be an original unquantized checkpoint')
    weights=sorted(directory.glob('*.safetensors'))
    if not weights:
        raise ValueError('source checkpoint has no safetensors weights')
    tokens=sorted(p for p in directory.iterdir() if p.name.startswith(('tokenizer','vocab','merges','added_tokens','special_tokens','chat_template')) and p.is_file())
    identity=dict(config_sha256=file_hash(directory/'config.json'),
                  weights=[dict(file=p.name,sha256=file_hash(p)) for p in weights],
                  tokenizer=[dict(file=p.name,sha256=file_hash(p)) for p in tokens])
    return dict(source_identity=digest(identity),source_files=identity)


def save_source_fingerprint(source, output):
    value=source_fingerprint(source)
    (Path(output)/'macaq_source.json').write_text(json.dumps(value,indent=2)+'\n')
