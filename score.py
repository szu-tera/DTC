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
from dtc.divergence import (  # noqa: E402
    dual_token_divergence_metrics,
    sequential_dual_metrics,
)
from dtc.io_utils import dump_jsonl, load_jsonl  # noqa: E402
from dtc.models import load_hf_on_cuda, release_cuda  # noqa: E402
from dtc.resolve_ids import resolve_ids  # noqa: E402


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


def _attach_dtc_block(row, metrics, *, boxed_idx):
    row["dtc"] = {
        "per_token": metrics.get("per_token", []),
        "boxed_token_indices": boxed_idx,
        "log_base": 2,
    }


def _boxed_indices(tok, completion_ids, generate: str):
    if completion_ids:
        surface = tok.decode(list(completion_ids), skip_special_tokens=True)
    else:
        surface = generate or ""
    return boxed_answer_positions_in_completion(tok, surface)


def run_whitebox(rows, args, ref_path: str, other_path: str):
    import torch

    n_vis = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[info] visible GPUs={n_vis} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    tok = __import__("transformers").AutoTokenizer.from_pretrained(
        os.path.abspath(ref_path), trust_remote_code=True
    )

    if n_vis >= 2:
        m_ref, _ = load_hf_on_cuda(ref_path, dtype=args.dtype, cuda_index=0)
        m_other, _ = load_hf_on_cuda(other_path, dtype=args.dtype, cuda_index=1)
        for row in tqdm(rows, desc="DTC whitebox"):
            prompt_ids, completion_ids = resolve_ids(row, tok)
            metrics = dual_token_divergence_metrics(
                m_ref,
                m_other,
                prompt_token_ids=prompt_ids,
                completion_token_ids=completion_ids,
                tokenizer=tok,
            )
            boxed_idx = _boxed_indices(tok, completion_ids, row.get("generate") or "")
            _attach_dtc_block(row, metrics, boxed_idx=boxed_idx)
        del m_ref, m_other
        release_cuda()
    else:
        chunk: list = []
        id_pairs: list = []
        for row in tqdm(rows, desc="Prep ids"):
            prompt_ids, completion_ids = resolve_ids(row, tok)
            chunk.append((prompt_ids, completion_ids))
            id_pairs.append(row)
        for i in range(0, len(chunk), 8):
            sub = chunk[i : i + 8]
            metrics_list = sequential_dual_metrics(
                sub, ref_path, other_path, tok, args.dtype
            )
            for row, (prompt_ids, completion_ids), metrics in zip(
                id_pairs[i : i + 8], sub, metrics_list
            ):
                boxed_idx = _boxed_indices(tok, completion_ids, row.get("generate") or "")
                _attach_dtc_block(row, metrics, boxed_idx=boxed_idx)


def run_blackbox(rows, args, big_path: str, small_path: str, tok_path: str):
    import torch

    n_vis = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[info] visible GPUs={n_vis} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    tok = __import__("transformers").AutoTokenizer.from_pretrained(
        os.path.abspath(tok_path), trust_remote_code=True
    )
    ref_path, other_path = big_path, small_path

    if n_vis >= 2:
        m_ref, _ = load_hf_on_cuda(ref_path, dtype=args.dtype, cuda_index=0)
        m_other, _ = load_hf_on_cuda(other_path, dtype=args.dtype, cuda_index=1)
        for row in tqdm(rows, desc="DTC blackbox"):
            prompt_ids, completion_ids = resolve_ids(row, tok)
            metrics = dual_token_divergence_metrics(
                m_ref,
                m_other,
                prompt_token_ids=prompt_ids,
                completion_token_ids=completion_ids,
                tokenizer=tok,
            )
            boxed_idx = _boxed_indices(tok, completion_ids, row.get("generate") or "")
            _attach_dtc_block(row, metrics, boxed_idx=boxed_idx)
            row["dtc_prob_key"] = "prob_ref"
        del m_ref, m_other
        release_cuda()
    else:
        for i in range(0, len(rows), 8):
            sub_rows = rows[i : i + 8]
            chunk = [resolve_ids(r, tok) for r in sub_rows]
            metrics_list = sequential_dual_metrics(
                chunk, ref_path, other_path, tok, args.dtype
            )
            for row, (prompt_ids, completion_ids), metrics in zip(sub_rows, chunk, metrics_list):
                boxed_idx = _boxed_indices(tok, completion_ids, row.get("generate") or "")
                _attach_dtc_block(row, metrics, boxed_idx=boxed_idx)
                row["dtc_prob_key"] = "prob_ref"


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
        tok_src = resolve_model_path(args.tokenizer_model or big)
        run_blackbox(rows, args, big, small, tok_src)
        for row in rows:
            row["dtc_setting"] = "blackbox"
            row["dtc_theta"] = THETA_BLACKBOX

    dump_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
