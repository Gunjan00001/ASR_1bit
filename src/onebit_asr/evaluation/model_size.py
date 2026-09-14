"""Model size accounting (FP32 baseline; binary packing comes in Phase 16)."""

from pathlib import Path

import torch.nn as nn
from huggingface_hub import snapshot_download


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def fp32_weight_bytes(total_params: int) -> int:
    return total_params * 4


def checkpoint_size_bytes(repo_id: str) -> tuple[int, list[str]]:
    """Size of the downloaded checkpoint weight files (safetensors/bin)."""
    snap = Path(snapshot_download(repo_id))
    files = sorted(
        str(p) for p in snap.rglob("*")
        if p.is_file() and p.suffix in (".safetensors", ".bin")
    )
    return sum(Path(f).stat().st_size for f in files), files
