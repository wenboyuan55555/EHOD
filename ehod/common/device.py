from __future__ import annotations

import os

import torch


REQUIRED_DEVICE = "cuda:0"


def require_cuda(device_name: str) -> torch.device:
    requested = os.environ.get("EHOD_CUDA_DEVICE", device_name)
    if requested == "auto":
        requested = "cuda:0"
    if not requested.startswith("cuda:"):
        raise ValueError("EHOD requires a CUDA device such as cuda:0; got %r" % requested)
    try:
        index = int(requested.split(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError("Invalid CUDA device %r" % requested) from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() <= index:
        raise RuntimeError(
            "%s is unavailable; visible CUDA device count=%d"
            % (requested, torch.cuda.device_count())
        )
    torch.cuda.set_device(index)
    return torch.device(requested)
