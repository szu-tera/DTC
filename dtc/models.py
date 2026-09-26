from __future__ import annotations

import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_hf(
    path: str,
    *,
    dtype: str = "bf16",
    device_map: object = "auto",
):
    tp = dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)[dtype]
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        dtype=tp,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tok


def load_hf_on_cuda(path: str, *, dtype: str, cuda_index: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    return load_hf(path, dtype=dtype, device_map={"": f"cuda:{cuda_index}"})


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
