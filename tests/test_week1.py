import pytest
import torch
import torch.nn.functional as F

from attnq.attention import attention
from attnq.quant import quantize_int8, dequantize

torch.manual_seed(0)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(1, 1, 4, 8), (2, 4, 16, 32), (3, 2, 7, 64)])
def test_matches_sdpa(shape, causal):
    q, k, v = (torch.randn(shape) for _ in range(3))
    torch.testing.assert_close(attention(q, k, v, causal=causal),
                               F.scaled_dot_product_attention(q, k, v, is_causal=causal))


def test_bool_mask_matches_sdpa():
    q, k, v = (torch.randn(2, 4, 10, 16) for _ in range(3))
    mask = torch.rand(10, 10) > 0.3
    mask[:, 0] = True  # every row needs one visible key, else softmax gives NaN
    torch.testing.assert_close(attention(q, k, v, mask=mask),
                               F.scaled_dot_product_attention(q, k, v, attn_mask=mask))


def test_int8_roundtrip_error_bounded():
    x = torch.randn(64, 128)
    q, scale = quantize_int8(x)
    assert q.dtype == torch.int8
    # rounding error is at most half a quantisation step per element
    assert ((dequantize(q, scale) - x).abs() <= scale / 2 + 1e-6).all()


def test_int8_zero_row_is_safe():
    q, scale = quantize_int8(torch.zeros(3, 8))
    assert torch.isfinite(dequantize(q, scale)).all()
