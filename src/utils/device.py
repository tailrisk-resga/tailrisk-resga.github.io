from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceConfig:
    device: torch.device
    use_gpu: bool
    cuda: int


def resolve_device(requested: str = "auto", gpu_id: int = 0) -> DeviceConfig:
    """Resolve a portable training device from config/CLI values.

    The legacy trainer currently supports CUDA and CPU. MPS is intentionally not
    selected here until the training loop is audited for Apple Silicon behavior.
    """

    requested = (requested or "auto").lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of {'auto', 'cpu', 'cuda'}")

    if requested == "cpu":
        return DeviceConfig(device=torch.device("cpu"), use_gpu=False, cuda=0)

    if torch.cuda.is_available() and requested in {"auto", "cuda"}:
        count = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= count:
            raise ValueError(f"gpu_id={gpu_id} is invalid; available CUDA devices: {count}")
        return DeviceConfig(device=torch.device(f"cuda:{gpu_id}"), use_gpu=True, cuda=gpu_id)

    if requested == "cuda":
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    return DeviceConfig(device=torch.device("cpu"), use_gpu=False, cuda=0)


def describe_runtime() -> dict[str, str | int | bool]:
    return {
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
