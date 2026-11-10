"""Select the integration matching the installed vLLM API."""

from importlib.metadata import version
import os


def register():
    release = tuple(int(part) for part in version('vllm').split('.')[:2])
    if release >= (0, 30):
        from macaq_env import configure_host_allocator
        configure_host_allocator(os.environ, tuple(int(part) for part in version('torch').split('.')[:2]))
        from macaq_modern_plugin import register as modern
        modern()
    else:
        import vllm_macaq_mxfp4_plugin
        vllm_macaq_mxfp4_plugin.register()
