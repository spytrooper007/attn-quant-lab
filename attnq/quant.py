import torch


def quantize_int8(x, dim=-1):
    """Symmetric per-channel int8 along `dim`. Returns (int8 tensor, float scale)."""
    scale = x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8) / 127
    return (x / scale).round().clamp(-127, 127).to(torch.int8), scale


def dequantize(q, scale):
    return q.float() * scale
