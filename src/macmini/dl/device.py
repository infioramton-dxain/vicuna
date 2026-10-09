""
from __future__ import annotations


def require_cuda(context: str = "model compute") -> str:
    ""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"GPU-only policy: CUDA unavailable for {context} — refusing "
            "CPU fallback. Check driver, "
            "VRAM pressure from concurrent tenants (other GPU tenants), or run "
            "inside dl_venv.")
    return "cuda:0"
