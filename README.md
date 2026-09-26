<div align="center">

# Probability is Not Enough: Exploring and Counting Divergent Tokens for Reasoning Uncertainty Quantification in LLMs

**Anonymous ICLR 2027 submission**

</div>

---

## Overview

**DTC (Divergent Token Confidence)** quantifies uncertainty in long chain-of-thought reasoning by counting tokens where next-token distributions disagree between a generator and a smaller auxiliary model (JSD above a threshold θ). Two path-level scores are reported:

- **DTClin** — linear map from the divergent-token count *m*
- **DTCprod** — trajectory-mean token probability raised to (*m* + *k*)

This repository contains a **minimal** pipeline to reproduce **DTC-only** results from the main paper tables: white-box Table 1 (ECE) and black-box Default CoT (ECE / AUROC). Baselines (NSL, UQAC, PRM, verbalized prompts, etc.) are not included.

---

## Getting Started

### 1. Environment

Requires [Conda](https://docs.conda.io/) (Miniconda or Anaconda). GPU drivers and CUDA compatible with your `torch` / `vllm` wheels are assumed on the host.

```bash
cd DTC
conda env create -f environment.yml
conda activate dtc
pip install -e evaluation/Qwen2.5-Math/evaluation/latex2sympy \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

`environment.yml` installs pip packages from the Tsinghua PyPI mirror. To refresh dependencies after pulling updates:

```bash
conda activate dtc
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e evaluation/Qwen2.5-Math/evaluation/latex2sympy \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Place model weights under `MODELS_DIR` (default `/path/to/models`) or pass full paths to the scripts. Run all commands below with `conda activate dtc` active.

```bash
export MODELS_DIR=/path/to/models
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
```

### 2. White-box example (Qwen2.5-7B-Instruct, MATH-500)

```bash
# Sample CoT trajectories (8 per question)
python sample.py \
  --model Qwen2.5-7B-Instruct \
  --dataset math500 \
  --cuda 0 \
  --overwrite

# Teacher-forcing JSD + token probs (generator + 1.5B aux)
python score.py \
  --input outputs/sample/math500_Qwen2.5-7B-Instruct.jsonl \
  --setting whitebox \
  --generator Qwen2.5-7B-Instruct \
  --aux Qwen2.5-1.5B-Instruct \
  --cuda 0,1 \
  --overwrite

# Metrics (balanced ECE / AUROC, 5 passes)
python eval.py \
  --input outputs/sample/math500_Qwen2.5-7B-Instruct_scored.jsonl
```

Operating θ for Qwen2.5 is **0.50** (Qwen3: 0.60, Gemma3-IT: 0.85). DTClin uses *a*=0.95, *b*=0.05, *n*=10; DTCprod uses *k*=4.

### 3. Black-box example (DeepSeek-V3.2, AIME24, Default CoT)

```bash
export DASHSCOPE_API_KEY=your_key

python sample.py \
  --model deepseek-v3.2 \
  --dataset aime24 \
  --backend api \
  --overwrite

python score.py \
  --input outputs/sample/aime24_deepseek-v3.2.jsonl \
  --setting blackbox \
  --big Qwen2.5-7B-Instruct \
  --small Qwen2.5-1.5B-Instruct \
  --cuda 0,1 \
  --overwrite

python eval.py --input outputs/sample/aime24_deepseek-v3.2_scored.jsonl
```

Black-box scoring uses **Qwen2.5-7B-Instruct** and **Qwen2.5-1.5B-Instruct** on the frozen trajectory (θ=**0.70**). The API generator is not loaded for confidence.

For local black-box generators (e.g. Qwen3-30B-A3B-Instruct-2507), use `--backend vllm` and the same `score.py` / `eval.py` steps.

---

## Pipeline

| Step | Script | Role |
|------|--------|------|
| Sample | `sample.py` | Zero-shot CoT generation + math grading |
| Score | `score.py` | Per-token JSD and probabilities (teacher forcing) |
| Eval | `eval.py` | Acc, DTClin / DTCprod ECE & AUROC |

Benchmark data and graders live under `evaluation/Qwen2.5-Math/evaluation/` (from [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math)). `evaluation/manual_eval/` is included for reference only.

Hyperparameters (decoding, samples per question, θ by model family) are centralized in `dtc/config.py`.

---

## Acknowledgements

- [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math) — data loading, answer extraction, and `math_equal` grading  
- [vLLM](https://github.com/vllm-project/vllm) — efficient local inference  

---

## Citation

```bibtex
@inproceedings{anonymous2027dtc,
  title={Probability is Not Enough: Exploring and Counting Divergent Tokens for Reasoning Uncertainty Quantification in LLMs},
  author={Anonymous},
  booktitle={Submitted to ICLR},
  year={2027}
}
```

Results depend on sampling randomness; with the same protocol (balanced UQ eval, truncated / no-answer filtering, θ and *n*, *k* as above), metrics should be close to the paper. Exact reproduction of every table cell is not guaranteed without matching hardware and full-scale runs.
