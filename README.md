# attn-quant-lab

From-scratch attention in PyTorch, then int8/int4 quantisation of it, measured on a real model.
CPU-first (Windows, Python 3.14, torch 2.15 CPU wheel). No Triton.

## Setup

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install numpy pytest
.\.venv\Scripts\python -m pytest -q
```

For an NVIDIA GPU later, swap the index URL for a CUDA one from pytorch.org and pass `device="cuda"`.

## Week 1 (done)

- `attnq/attention.py`: `attention(q, k, v, mask, causal)`, checked against `F.scaled_dot_product_attention`.
- `attnq/quant.py`: symmetric per-channel int8 `quantize_int8` / `dequantize`.
- `tests/test_week1.py`: 9 tests (shapes, causal, bool mask, int8 error bound, all-zero rows).

## Week 2: quantised attention + int4 (you write this)

1. `quantized_attention(q, k, v, bits=8)`: quantise K and V (per-head, per-channel), dequantise, call `attention`.
   Test: output error vs fp32 shrinks as bits go 4 -> 8.
2. `quantize_int4`: same as int8 but range [-7, 7]. Then pack two int4 values into one uint8 (`(a & 0xF) | (b << 4)`) and write `unpack`. Test the round trip exactly.
3. Try group-wise scales (one scale per 32 or 64 values) for int4 and show it beats per-channel. This is what GPTQ/AWQ-style kernels actually do.
4. Plug into GPT-2: `pip install transformers datasets`, load `gpt2`, monkeypatch its attention to call yours,
   and compute WikiText-2 perplexity for fp32 / int8 / int4. Expect fp32 around 29-30 at stride 1024.

## Week 3: benchmarks + charts

1. `bench.py`: for seq lengths 128..4096, time fp32 vs int8 vs int4 attention (`torch.utils.benchmark.Timer`),
   record KV-cache bytes (`tensor.element_size() * numel()`), and max/mean abs error. Save to `results/*.csv`.
2. `plot.py`: matplotlib charts of latency, memory, and perplexity vs bits. Put the PNGs in this README.
3. Write a "Results" section: one table, three charts, two sentences on the trade-off you found.

## Bonus: sweep agent

A script that tries combinations (bits x group size x which layers are quantised), runs perplexity for each,
and reports the Pareto front of memory vs perplexity. Start as a plain loop; only add an LLM in the loop
if the plain loop is too slow to search the space.
