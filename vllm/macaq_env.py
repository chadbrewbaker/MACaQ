"""Runtime environment for text serving and evaluation.

Call setup_cuda_env() before importing tensor or serving libraries. Worker
processes inherit compilation settings and host allocation defaults. Explicit
user settings are preserved. Numeric compute capabilities select compatible
compiler targets; an installed toolkit is added to the search path when present.
"""
import os
import json
import subprocess
import sys


def configure_host_allocator(environ, torch_release):
    """Avoid oversized pinned tables while respecting explicit allocator tuning."""
    if torch_release < (2, 13):
        return
    key = ('PYTORCH_ALLOC_CONF' if 'PYTORCH_ALLOC_CONF' in environ
           else 'PYTORCH_CUDA_ALLOC_CONF' if 'PYTORCH_CUDA_ALLOC_CONF' in environ
           else 'PYTORCH_ALLOC_CONF')
    value = environ.get(key, '')
    for option in ('pinned_max_round_threshold_mb', 'pinned_max_cached_size_mb'):
        if option + ':' not in ''.join(value.split()):
            value = ','.join(filter(None, (value, option + ':64')))
    environ[key] = value


def _compute_caps():
    """Read numeric capabilities without initializing the serving process."""
    query = (
        "import json, torch; "
        "print(json.dumps(sorted({"
        "'.'.join(map(str, torch.cuda.get_device_capability(i))) "
        "for i in range(torch.cuda.device_count())})))"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", query], capture_output=True, text=True,
            timeout=30, check=True).stdout
        caps = json.loads(out)
    except (subprocess.SubprocessError, OSError, ValueError):
        return []
    if not isinstance(caps, list) or not all(
            isinstance(cap, str) and len(cap.split(".")) == 2
            and all(part.isdigit() for part in cap.split(".")) for cap in caps):
        return []
    return sorted(set(caps))


def _flashinfer_arch(cap: str) -> str:
    """Select FlashInfer's suffix from the numeric compute capability."""
    major = int(cap.split(".")[0])
    if major in (9, 10):
        return cap + "a"
    if major in (11, 12):
        return cap + "f"
    return cap


def setup_cuda_env(cuda_home_default="/usr/local/cuda"):
    """Configure compiler and host allocation defaults; return capabilities."""
    from importlib.metadata import version, PackageNotFoundError
    try:
        torch_release = tuple(int(part) for part in version('torch').split('.')[:2])
    except PackageNotFoundError:
        torch_release = (0, 0)
    configure_host_allocator(os.environ, torch_release)
    caps = _compute_caps()
    if caps:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST",
                              " ".join(f"{c}+PTX" for c in caps))
        os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST",
                              " ".join(_flashinfer_arch(c) for c in caps))

    cuda_home = os.environ.get("CUDA_HOME") or cuda_home_default
    if os.path.isdir(os.path.join(cuda_home, "bin")):
        os.environ.setdefault("CUDA_HOME", cuda_home)
        os.environ["PATH"] = f"{cuda_home}/bin:{os.environ.get('PATH', '')}"
        ptxas = os.path.join(cuda_home, "bin", "ptxas")
        try:
            compiler_release = tuple(int(part) for part in version('triton').split('.')[:2])
        except PackageNotFoundError:
            compiler_release = (0, 0)
        if os.path.isfile(ptxas) and compiler_release < (3, 7):
            os.environ.setdefault("TRITON_PTXAS_PATH", ptxas)

    return caps


if __name__ == "__main__":
    print("compute caps:", _compute_caps() or "(none detected)")
    setup_cuda_env()
    for k in ("TORCH_CUDA_ARCH_LIST", "FLASHINFER_CUDA_ARCH_LIST",
              "CUDA_HOME", "TRITON_PTXAS_PATH"):
        print(f"{k}={os.environ.get(k)}")
