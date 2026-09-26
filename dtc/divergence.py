from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def forward_completion_logits(
    model,
    *,
    prompt_token_ids: List[int],
    completion_token_ids: List[int],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Teacher forcing: logits predicting each completion token, shape [T, V]."""
    if not completion_token_ids:
        return torch.empty(0, device=device or _model_device(model))
    dev = device or _model_device(model)
    p = torch.tensor([prompt_token_ids], dtype=torch.long, device=dev)
    c = torch.tensor([completion_token_ids], dtype=torch.long, device=dev)
    full = torch.cat([p, c], dim=1)
    out = model(full)
    logits = out.logits
    p_len = len(prompt_token_ids)
    # position p_len-1 predicts c[0], ..., p_len+T-2 predicts c[T-1]
    sl = logits[0, p_len - 1 : p_len - 1 + len(completion_token_ids), :]
    return sl


def _kl_rowwise(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p + eps
    q = q + eps
    return (p * (p.log2() - q.log2())).sum(dim=-1)


@torch.inference_mode()
def dual_token_divergence_metrics(
    model_ref,
    model_other,
    *,
    prompt_token_ids: List[int],
    completion_token_ids: List[int],
    tokenizer=None,
    compute_device: Optional[torch.device] = None,
) -> Dict:
    lr = forward_completion_logits(
        model_ref,
        prompt_token_ids=prompt_token_ids,
        completion_token_ids=completion_token_ids,
    )
    lo = forward_completion_logits(
        model_other,
        prompt_token_ids=prompt_token_ids,
        completion_token_ids=completion_token_ids,
    )
    return dual_metrics_from_logits(
        lr,
        lo,
        completion_token_ids=completion_token_ids,
        tokenizer=tokenizer,
        compute_device=compute_device,
    )


@torch.inference_mode()
def dual_metrics_from_logits(
    logits_ref: torch.Tensor,
    logits_other: torch.Tensor,
    *,
    completion_token_ids: List[int],
    tokenizer=None,
    compute_device: Optional[torch.device] = None,
) -> Dict:
    if not completion_token_ids:
        return {"per_token": []}

    seq_len = min(int(logits_ref.size(0)), int(logits_other.size(0)), len(completion_token_ids))
    dev = compute_device or (logits_ref.device if logits_ref.is_cuda else torch.device("cpu"))
    if torch.cuda.is_available() and dev.type != "cuda":
        dev = torch.device("cuda", torch.cuda.current_device())

    vocab = min(int(logits_ref.size(-1)), int(logits_other.size(-1)))
    eps = 1e-12
    per_token: List[Dict] = []

    chunk = 256
    for start in range(0, seq_len, chunk):
        end = min(start + chunk, seq_len)
        lr = logits_ref[start:end, :vocab].to(device=dev, dtype=torch.float32)
        lo = logits_other[start:end, :vocab].to(device=dev, dtype=torch.float32)
        p_ref = F.softmax(lr, dim=-1)
        p_other = F.softmax(lo, dim=-1)
        m = 0.5 * (p_ref + p_other)
        js = 0.5 * _kl_rowwise(p_ref, m, eps) + 0.5 * _kl_rowwise(p_other, m, eps)

        tok_ids = torch.as_tensor(
            completion_token_ids[start:end], dtype=torch.long, device=dev
        ).clamp(min=0, max=max(vocab - 1, 0))
        prob_ref = p_ref.gather(1, tok_ids.unsqueeze(1)).squeeze(1)
        prob_other = p_other.gather(1, tok_ids.unsqueeze(1)).squeeze(1)

        js_cpu = js.detach().cpu()
        pr_cpu = prob_ref.detach().cpu()
        po_cpu = prob_other.detach().cpu()
        for i in range(end - start):
            row = {
                "index": start + i,
                "js_divergence": float(js_cpu[i].item()),
                "prob_ref": float(pr_cpu[i].item()),
                "prob_other": float(po_cpu[i].item()),
            }
            if tokenizer is not None:
                row["token_str"] = tokenizer.decode(
                    [int(completion_token_ids[start + i])], skip_special_tokens=False
                )
            per_token.append(row)
        del lr, lo, p_ref, p_other, m, js, prob_ref, prob_other

    return {"per_token": per_token}
