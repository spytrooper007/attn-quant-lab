import math
import torch


def attention(q, k, v, mask=None, causal=False):
    """softmax(QK^T / sqrt(d)) V. q,k,v: (..., seq, d). mask: bool, True = keep."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if causal:
        L, S = q.size(-2), k.size(-2)
        mask = torch.ones(L, S, dtype=torch.bool, device=q.device).tril()
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(-1) @ v
