import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
from datasets import Dataset

STORAGE_PATH = os.getenv("STORAGE_PATH")
if not STORAGE_PATH:
    raise RuntimeError("STORAGE_PATH environment variable must be set before running upload.py")

GENERATED_DIR = Path(STORAGE_PATH) / "generated_question"
DATASET_ROOT = Path(STORAGE_PATH) / "datasets"


def load_results(experiment_name: str) -> List[Dict]:
    datas: List[Dict] = []
    for i in range(8):
        result_path = GENERATED_DIR / f"{experiment_name}_{i}_results.json"
        try:
            with result_path.open("r") as f:
                datas.extend(json.load(f))
        except FileNotFoundError:
            print(f"File {result_path.name} not found")

    for i in range(8):
        result_path = GENERATED_DIR / f"{experiment_name}_{i}_results.json"
        try:
            result_path.unlink()
        except FileNotFoundError:
            continue

    if not datas:
        raise RuntimeError("No evaluation results found. Ensure question_evaluate/evaluate.py finished successfully.")

    return datas


def plot_score_histogram(scores: List[float], experiment_name: str) -> None:
    if not scores:
        return
    plt.hist(scores, bins=11)
    plt.title(f"Score distribution: {experiment_name}")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.savefig(f"scores_distribution_{experiment_name}.png")
    plt.close()


def build_parquet_dataset(records: List[Dict], experiment_name: str) -> Path:
    dataset_dir = DATASET_ROOT / experiment_name / "train"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = dataset_dir / "data.parquet"

    dataset = Dataset.from_list(records)
    if parquet_path.exists():
        parquet_path.unlink()
    dataset.to_parquet(str(parquet_path))

    stats = {
        "num_samples": len(records),
        "min_score": min(record["score"] for record in records),
        "max_score": max(record["score"] for record in records),
        "mean_score": sum(record["score"] for record in records) / len(records),
    }
    stats_path = dataset_dir.parent / "stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    return dataset_dir.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_score", type=float, default=0.7)
    parser.add_argument("--min_score", type=float, default=0.3)
    parser.add_argument("--experiment_name", type=str, default="Qwen_Qwen3-4B-Base_all")
    args = parser.parse_args()

    datas = load_results(args.experiment_name)

    scores = [data["score"] for data in datas if "score" in data]
    plot_score_histogram(scores, args.experiment_name)

    filtered_datas = [
        {
            "problem": data["question"],
            "answer": data["answer"],
            "score": data["score"],
        }
        for data in datas
        if data.get("score") is not None
        and args.min_score <= data["score"] <= args.max_score
        and data.get("answer")
        and data.get("answer") != "None"
    ]

    if not filtered_datas:
        raise RuntimeError(
            "No samples satisfied the score filters. Try widening the score range or rerunning generation."
        )

    dataset_dir = build_parquet_dataset(filtered_datas, args.experiment_name)
    print(f"Saved local dataset for {args.experiment_name} to {dataset_dir}")


if __name__ == "__main__":
    main()
