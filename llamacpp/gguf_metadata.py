"""GGUF tokenizer metadata and runtime fingerprints."""
from functools import lru_cache
import hashlib
import re
import subprocess
import sys
from pathlib import Path


def _file_identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=64)
def _hash_unchanged_file(path, identity):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def sha256(path):
    """Reuse verified digests within a run while detecting file mutations."""
    path = Path(path).resolve()
    identity = _file_identity(path)
    digest = _hash_unchanged_file(path, identity)
    if _file_identity(path) != identity:
        raise ValueError('File changed while computing its fingerprint')
    return digest


def runtime_libraries(binary):
    """Record linked runtime libraries: an executable hash alone is insufficient."""
    if not sys.platform.startswith('linux'):
        return None
    run = subprocess.run(['ldd', str(binary.resolve())], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, check=False)
    output = run.stdout
    if run.returncode:
        if 'not a dynamic executable' in output or 'statically linked' in output:
            return {}
        raise ValueError(f'Cannot inspect linked runtime libraries: {output.strip()}')
    libraries = {}
    for line in output.splitlines():
        if re.match(r'\s*lib(?:llama|ggml|cublas|cudart)', line) and 'not found' in line:
            raise ValueError(f'Missing runtime library: {line.strip()}')
        match = re.match(r'\s*(lib(?:llama|ggml|cublas|cudart)[^\s]*)\s+=>\s+(/\S+)', line)
        if match:
            path = Path(match[2]).resolve()
            libraries[match[1]] = dict(path=str(path), sha256=sha256(path))
    return libraries


def serializable(value):
    if hasattr(value, "tolist"):
        return serializable(value.tolist())
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def input_tokenizer_metadata(tokenizer):
    # Raw, unpadded scoring never uses padding or a chat template. A BOS ID
    # is likewise inactive only when both encodings explicitly disable BOS.
    encoding = {k: v for k, v in tokenizer.items()
                if not k.startswith('tokenizer.chat_template')
                and k != 'tokenizer.ggml.padding_token_id'}
    if encoding.get('tokenizer.ggml.add_bos_token') is False:
        encoding.pop('tokenizer.ggml.bos_token_id', None)
    return encoding
