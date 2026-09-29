#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _cuda_from_argv() -> None:
    """Must run before torch is imported, otherwise the visible GPU set is ignored."""
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--cuda" and i + 1 < len(argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = argv[i + 1]
            return
        if arg.startswith("--cuda="):
            os.environ["CUDA_VISIBLE_DEVICES"] = arg.split("=", 1)[1]
            return


_cuda_from_argv()

from tqdm import tqdm

from dtc.boxed import boxed_answer_positions_in_completion  # noqa: E402
from dtc.config import (  # noqa: E402
    THETA_BLACKBOX,
    resolve_model_path,
    whitebox_theta,
)
from dtc.divergence import dual_token_divergence_metrics  # noqa: E402
from dtc.io_utils import dump_jsonl, load_jsonl  # noqa: E402
from dtc.models import load_hf_on_cuda, release_cuda  # noqa: E402
from dtc.resolve_ids import resolve_ids  # noqa: E402
from dtc.verbalized_spans import (  # noqa: E402
    resolve_verbalized_selected_answer_from_row,
    verbalized_reason_ans_token_indices,
)


def parse_args():
    p = argparse.ArgumentParser(description="Teacher-forcing JSD + token probs for DTC")
    p.add_argument("--input", required=True, help="Sample jsonl from sample.py")
    p.add_argument("--output", default=None)
    p.add_argument("--setting", choices=("whitebox", "blackbox"), required=True)
    p.add_argument("--generator", default=None, help="White-box: generating model path")
    p.add_argument("--aux", default=None, help="White-box: small auxiliary model path")
    p.add_argument("--big", default=None, help="Black-box: Qwen2.5-7B-Instruct path")
    p.add_argument("--small", default=None, help="Black-box: Qwen2.5-1.5B-Instruct path")
    p.add_argument("--tokenizer-model", default=None, help="Black-box: tokenizer source (default: big)")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--cuda", default="0,1", help="Visible GPUs for dual models")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def _attach_dtc_block(row, metrics, *, boxed_idx, span_source: str):
    row["dtc"] = {
        "per_token": metrics.get("per_token", []),
        "boxed_token_indices": boxed_idx,
        "log_base": 2,
    }
    row["dtc_answer_span_source"] = span_source


def _completion_surface(tok, completion_ids, generate: str) -> str:
    if completion_ids:
        return tok.decode(list(completion_ids), skip_special_tokens=True)
    return generate or ""


def _answer_span_indices(row, tok, completion_ids, generate: str):
    surface = _completion_surface(tok, completion_ids, generate)
    method, selected = resolve_verbalized_selected_answer_from_row(row)
    if method:
        _reason_idx, ans_idx = verbalized_reason_ans_token_indices(
            tok,
            surface,
            method=method,
            selected_answer=selected,
        )
        return ans_idx, "verbalized_json"
    return boxed_answer_positions_in_completion(tok, surface), "boxed"


def _load_pair(ref_path: str, other_path: str, dtype: str, n_vis: int):
    """Load each model once. Two GPUs: one card each. One GPU: both stay resident."""
    if n_vis >= 2:
        m_ref, _ = load_hf_on_cuda(ref_path, dtype=dtype, cuda_index=0)
        m_other, _ = load_hf_on_cuda(other_path, dtype=dtype, cuda_index=1)
    else:
        m_ref, _ = load_hf_on_cuda(ref_path, dtype=dtype, cuda_index=0)
        m_other, _ = load_hf_on_cuda(other_path, dtype=dtype, cuda_index=0)
    return m_ref, m_other


def _score_loaded(rows, tok, m_ref, m_other, *, desc: str):
    for row in tqdm(rows, desc=desc):
        prompt_ids, completion_ids = resolve_ids(row, tok)
        metrics = dual_token_divergence_metrics(
            m_ref,
            m_other,
            prompt_token_ids=prompt_ids,
            completion_token_ids=completion_ids,
            tokenizer=tok,
        )
        boxed_idx, span_source = _answer_span_indices(
            row, tok, completion_ids, row.get("generate") or ""
        )
        _attach_dtc_block(row, metrics, boxed_idx=boxed_idx, span_source=span_source)


def run_whitebox(rows, args, ref_path: str, other_path: str):
    import torch

    n_vis = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[info] visible GPUs={n_vis} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    tok = __import__("transformers").AutoTokenizer.from_pretrained(
        os.path.abspath(ref_path), trust_remote_code=True
    )
    m_ref, m_other = _load_pair(ref_path, other_path, args.dtype, n_vis)
    _score_loaded(rows, tok, m_ref, m_other, desc="DTC whitebox")
    del m_ref, m_other
    release_cuda()


def run_blackbox(rows, args, big_path: str, small_path: str, tok_path: str):
    import torch

    n_vis = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[info] visible GPUs={n_vis} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    tok = __import__("transformers").AutoTokenizer.from_pretrained(
        os.path.abspath(tok_path), trust_remote_code=True
    )
    m_ref, m_other = _load_pair(big_path, small_path, args.dtype, n_vis)
    _score_loaded(rows, tok, m_ref, m_other, desc="DTC blackbox")
    for row in rows:
        row["dtc_prob_key"] = "prob_ref"
    del m_ref, m_other
    release_cuda()


def main():
    args = parse_args()
    in_path = Path(args.input)
    rows = load_jsonl(in_path)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("Empty input")

    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_scored.jsonl")
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Exists: {out_path}")

    if args.setting == "whitebox":
        if not args.generator or not args.aux:
            raise SystemExit("whitebox requires --generator and --aux")
        ref = resolve_model_path(args.generator)
        other = resolve_model_path(args.aux)
        run_whitebox(rows, args, ref, other)
        gen_name = Path(ref).name
        row_theta = whitebox_theta(gen_name)
        for row in rows:
            row["dtc_setting"] = "whitebox"
            row["dtc_theta"] = row_theta
            row["dtc_prob_key"] = "prob_ref"
    else:
        big = resolve_model_path(args.big or "Qwen2.5-7B-Instruct")
        small = resolve_model_path(args.small or "Qwen2.5-1.5B-Instruct")
        tok_default = big
        if not args.tokenizer_model and rows:
            instruct = rows[0].get("instruct_model_path")
            if isinstance(instruct, str) and instruct.strip():
                p = Path(instruct)
                if p.is_dir() or p.exists():
                    tok_default = instruct
        tok_src = resolve_model_path(args.tokenizer_model or tok_default)
        run_blackbox(rows, args, big, small, tok_src)
        for row in rows:
            row["dtc_setting"] = "blackbox"
            row["dtc_theta"] = THETA_BLACKBOX

    dump_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
