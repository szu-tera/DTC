#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dtc.config import (  # noqa: E402
    OUTPUTS_DIR,
    resolve_model_path,
    sampling_params,
)
from dtc.eval_grader import grade_rows, load_dataset_examples  # noqa: E402
from dtc.io_utils import dump_jsonl  # noqa: E402
from dtc.sampling_utils import (  # noqa: E402
    apply_chat,
    build_stop_words,
    default_messages,
    resolve_stop_token_ids,
    strip_think_block,
)


def parse_args():
    p = argparse.ArgumentParser(description="Sample CoT trajectories for DTC evaluation")
    p.add_argument("--model", required=True, help="Generator model path or HF id")
    p.add_argument("--dataset", required=True, help="e.g. math500, aime24, hmmt_25")
    p.add_argument("--backend", choices=("vllm", "api"), default="vllm")
    p.add_argument("--output", default=None, help="Output jsonl (default: outputs/sample/...)")
    p.add_argument("--limit", type=int, default=0, help="Max problems (smoke test)")
    p.add_argument("--num-samples", type=int, default=None, help="Override samples per question")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES")
    p.add_argument("--api-base", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    p.add_argument("--api-key", default=None, help="Default: env DASHSCOPE_API_KEY")
    p.add_argument("--api-model", default=None, help="API model name if different from basename")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _default_output(model: str, dataset: str) -> Path:
    from dtc.config import _basename

    name = _basename(model)
    return OUTPUTS_DIR / "sample" / f"{dataset}_{name}.jsonl"


def run_vllm(args, out_path: Path, sp, gen_path: str):
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    ensure_eval = __import__("dtc.eval_grader", fromlist=["ensure_eval_imports"]).ensure_eval_imports
    ensure_eval()
    from parser import extract_answer, parse_question  # type: ignore
    from utils import set_seed  # type: ignore

    set_seed(args.seed)
    raw = load_dataset_examples(args.dataset, limit=args.limit)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok_dbg = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True)
    prompts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for ex in raw:
        ex = dict(ex)
        q = parse_question(ex, args.dataset).strip()
        if not q:
            continue
        prompt = apply_chat(tok_dbg, default_messages(q), enable_thinking=False)
        prompts.append(prompt)
        ex["question"] = q
        metas.append(ex)

    if not prompts:
        print("No prompts")
        return

    llm = LLM(model=gen_path, trust_remote_code=True, tensor_parallel_size=1)
    tokenizer = llm.get_tokenizer()
    stop_ids = resolve_stop_token_ids(tokenizer, model_dir=gen_path)
    stops = build_stop_words()
    print(f"[info] stop_token_ids={stop_ids}", flush=True)

    n = sp.num_samples
    per = min(16, n)
    rounds = math.ceil(n / per)
    rows: List[Dict[str, Any]] = []
    global_idx = 0
    t0 = time.time()

    for r in range(rounds):
        this_n = per if r < rounds - 1 else (n - per * (rounds - 1))
        sp_kwargs: Dict[str, Any] = dict(
            temperature=sp.temperature,
            top_p=sp.top_p,
            max_tokens=sp.max_tokens,
            n=this_n,
            stop=stops,
            stop_token_ids=stop_ids,
            seed=args.seed + r,
        )
        if sp.top_k > 0:
            sp_kwargs["top_k"] = sp.top_k
        if sp.presence_penalty != 0.0:
            sp_kwargs["presence_penalty"] = sp.presence_penalty
        sampling = SamplingParams(**sp_kwargs)
        outputs = llm.generate(prompts, sampling)

        for bi, req in enumerate(outputs):
            meta = metas[bi]
            dataset_id = meta.get("unique_id", meta.get("id", meta.get("idx", bi)))
            count_for = len([x for x in rows if x.get("dataset_id") == dataset_id])
            for comp in req.outputs:
                if count_for >= n:
                    break
                gen_text = (getattr(comp, "text", None) or "").rstrip()
                finish_reason = getattr(comp, "finish_reason", None)
                truncated = finish_reason == "length"
                token_ids = getattr(comp, "token_ids", None)
                prompt_ids = getattr(req, "prompt_token_ids", None)
                row = {
                    **dict(meta),
                    "dataset_id": dataset_id,
                    "row_idx": global_idx,
                    "sample_id": count_for,
                    "dataset": args.dataset,
                    "instruct_model_path": gen_path,
                    "prompt_text": req.prompt,
                    "prompt_token_ids": list(prompt_ids) if prompt_ids is not None else None,
                    "completion_token_ids": list(token_ids) if token_ids is not None else None,
                    "generate": gen_text,
                    "pred_answer_extracted": extract_answer(
                        strip_think_block(gen_text), args.dataset
                    ),
                    "correct": False,
                    "truncated": bool(truncated),
                    "finish_reason": finish_reason,
                    "max_tokens_per_call": sp.max_tokens,
                }
                rows.append(row)
                count_for += 1
                global_idx += 1

    grade_rows(rows, args.dataset)
    dump_jsonl(out_path, rows)
    print(f"Done: {len(rows)} rows -> {out_path} ({(time.time()-t0)/60:.1f} min)")


def run_api(args, out_path: Path, sp, gen_path: str):
    ensure_eval = __import__("dtc.eval_grader", fromlist=["ensure_eval_imports"]).ensure_eval_imports
    ensure_eval()
    from parser import extract_answer, parse_question  # type: ignore
    from utils import set_seed  # type: ignore

    set_seed(args.seed)
    raw = load_dataset_examples(args.dataset, limit=args.limit)

    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit("pip install openai for --backend api") from e

    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set DASHSCOPE_API_KEY or pass --api-key")

    from dtc.config import _basename

    api_model = args.api_model or _basename(args.model)
    client = OpenAI(api_key=api_key, base_url=args.api_base)

    rows: List[Dict[str, Any]] = []
    global_idx = 0
    t0 = time.time()

    for ex in raw:
        ex = dict(ex)
        q = parse_question(ex, args.dataset).strip()
        if not q:
            continue
        dataset_id = ex.get("unique_id", ex.get("id", ex.get("idx", global_idx)))
        messages = default_messages(q)
        for sid in range(sp.num_samples):
            resp = client.chat.completions.create(
                model=api_model,
                messages=messages,
                temperature=sp.temperature,
                top_p=sp.top_p,
                max_tokens=sp.max_tokens,
            )
            gen_text = (resp.choices[0].message.content or "").rstrip()
            finish = getattr(resp.choices[0], "finish_reason", None)
            truncated = finish == "length"
            row = {
                **ex,
                "dataset_id": dataset_id,
                "row_idx": global_idx,
                "sample_id": sid,
                "dataset": args.dataset,
                "instruct_model_path": gen_path,
                "prompt_text": str(messages),
                "generate": gen_text,
                "pred_answer_extracted": extract_answer(
                    strip_think_block(gen_text), args.dataset
                ),
                "correct": False,
                "truncated": bool(truncated),
                "finish_reason": finish,
                "max_tokens_per_call": sp.max_tokens,
            }
            rows.append(row)
            global_idx += 1

    grade_rows(rows, args.dataset)
    dump_jsonl(out_path, rows)
    print(f"Done: {len(rows)} rows -> {out_path} ({(time.time()-t0)/60:.1f} min)")


def main():
    args = parse_args()
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    gen_path = resolve_model_path(args.model)
    sp = sampling_params(args.model, args.dataset, backend=args.backend)
    if args.num_samples is not None:
        from dataclasses import replace

        sp = replace(sp, num_samples=int(args.num_samples))
    out_path = Path(args.output) if args.output else _default_output(args.model, args.dataset)
    if out_path.exists() and not args.overwrite:
        print(f"Exists: {out_path} (use --overwrite)")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[info] dataset={args.dataset} backend={args.backend} "
        f"temp={sp.temperature} top_p={sp.top_p} max_tokens={sp.max_tokens} n={sp.num_samples}",
        flush=True,
    )
    if args.backend == "api":
        run_api(args, out_path, sp, gen_path)
    else:
        run_vllm(args, out_path, sp, gen_path)


if __name__ == "__main__":
    main()
