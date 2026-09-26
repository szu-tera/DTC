import os
import json
import random
import datasets
from datasets import load_dataset, Dataset, concatenate_datasets
from utils import load_jsonl, lower_keys


def load_data(data_name, split, data_dir="./data"):
    data_file = f"{data_dir}/{data_name}/{split}.jsonl"
    # hmmt_25 / hmmt_feb_2026 等：目录下用同名 jsonl 代替 test.jsonl
    if (
        data_name in ("hmmt_25", "hmmt_feb_2026")
        and split == "test"
        and not os.path.exists(data_file)
    ):
        alt_file = f"{data_dir}/{data_name}/{data_name}.jsonl"
        if os.path.exists(alt_file):
            data_file = alt_file
    if os.path.exists(data_file):
        examples = list(load_jsonl(data_file))
    else:
        if data_name == "math":
            dataset = load_dataset(
                "competition_math",
                split=split,
                name="main",
                cache_dir=f"{data_dir}/temp",
            )
        elif data_name == "math500":
            # 只取 math 测试集前 500 条，避免默认跑完整 5k
            dataset = load_dataset(
                "competition_math",
                split="test[:500]" if split == "test" else split,
                name="main",
                cache_dir=f"{data_dir}/temp",
            )
        elif data_name == "gsm8k":
            dataset = load_dataset(data_name, split=split)
        elif data_name == "svamp":
            # evaluate on training set + test set
            dataset = load_dataset("ChilleD/SVAMP", split="train")
            dataset = concatenate_datasets(
                [dataset, load_dataset("ChilleD/SVAMP", split="test")]
            )
        elif data_name == "asdiv":
            dataset = load_dataset("EleutherAI/asdiv", split="validation")
            dataset = dataset.filter(
                lambda x: ";" not in x["answer"]
            )  # remove multi-answer examples
        elif data_name == "mawps":
            examples = []
            # four sub-tasks
            for data_name in ["singleeq", "singleop", "addsub", "multiarith"]:
                sub_examples = list(load_jsonl(f"{data_dir}/mawps/{data_name}.jsonl"))
                for example in sub_examples:
                    example["type"] = data_name
                examples.extend(sub_examples)
            dataset = Dataset.from_list(examples)
        elif data_name == "mmlu_stem":
            dataset = load_dataset("hails/mmlu_no_train", "all", split="test")
            # only keep stem subjects
            stem_subjects = [
                "abstract_algebra",
                "astronomy",
                "college_biology",
                "college_chemistry",
                "college_computer_science",
                "college_mathematics",
                "college_physics",
                "computer_security",
                "conceptual_physics",
                "electrical_engineering",
                "elementary_mathematics",
                "high_school_biology",
                "high_school_chemistry",
                "high_school_computer_science",
                "high_school_mathematics",
                "high_school_physics",
                "high_school_statistics",
                "machine_learning",
            ]
            dataset = dataset.rename_column("subject", "type")
            dataset = dataset.filter(lambda x: x["type"] in stem_subjects)
        elif data_name == "carp_en":
            dataset = load_jsonl(f"{data_dir}/carp_en/test.jsonl")
        elif data_name in ("aime25", "aime26"):
            # math-ai/aime25、math-ai/aime26：字段 problem / answer / id
            dataset = load_dataset(
                f"math-ai/{data_name}",
                split=split,
                cache_dir=f"{data_dir}/temp",
            )
        elif data_name == "gpqa_diamond":
            import pandas as pd
            parquet_path = f"{data_dir}/gpqa_diamond/gpqa_diamond.parquet"
            if os.path.exists(parquet_path):
                df = pd.read_parquet(parquet_path)
                examples = [
                    {"idx": i, "question": str(row["question"]), "answer": str(row["answer"]).strip().upper()}
                    for i, row in df.iterrows()
                ]
                dataset = Dataset.from_list(examples)
            else:
                raise FileNotFoundError(f"gpqa_diamond 需 {parquet_path} 或 {data_file}")
        elif data_name in ("zebralogic_sm", "zebralogic", "zebra_sm", "zebra_grid"):
            # ZebraLogic grid_mode；公开 allenai 集 solution 已打码，改用 WildEval（含答案）
            # zebralogic_sm / zebra_sm / zebralogic 默认 Small+Medium；全量用 zebra_grid
            from BaseCal.src.sampling.zebra import (  # type: ignore
                build_user_prompt,
                filter_sm_examples,
                gold_solution_json,
                normalize_size,
            )

            raw = load_dataset(
                "WildEval/ZebraLogic",
                "grid_mode",
                split="test",
                cache_dir=f"{data_dir}/temp",
            )
            examples = [dict(x) for x in raw]
            for i, ex in enumerate(examples):
                ex["idx"] = i
                ex["size"] = normalize_size(ex.get("size"))
                ex["id"] = str(ex.get("id", i))
                sol = ex.get("solution")
                if isinstance(sol, str):
                    sol = json.loads(sol)
                    ex["solution"] = sol
                # answer = 规范化 House 表 JSON，供 puzzle 二值判分
                ex["answer"] = gold_solution_json(ex)
                # 校验：公开打码集全是 ___，拒绝写入坏缓存
                flat = [c for r in (sol or {}).get("rows", []) for c in r]
                if flat and all(str(c).strip() == "___" for c in flat):
                    raise RuntimeError(
                        "ZebraLogic solution 全为占位符 ___；请改用 "
                        "WildEval/ZebraLogic 或申请 allenai/ZebraLogicBench-private"
                    )
                ex["question"] = build_user_prompt(ex.get("puzzle", ""), sol or {})
            if data_name != "zebra_grid":
                examples = filter_sm_examples(examples)
                for i, ex in enumerate(examples):
                    ex["idx"] = i
            dataset = Dataset.from_list(examples)
        else:
            raise NotImplementedError(data_name)

        examples = list(dataset)
        examples = [lower_keys(example) for example in examples]
        dataset = Dataset.from_list(examples)
        os.makedirs(f"{data_dir}/{data_name}", exist_ok=True)
        dataset.to_json(data_file)

    # add 'idx' in the first column
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]

    # dedepulicate & sort
    examples = sorted(examples, key=lambda x: x["idx"])
    return examples
