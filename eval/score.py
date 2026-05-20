import os
import pandas as pd

RESULT_PATHS = {
    "finetuned": {
        "bllossom": "eval_results/bllossom_finetuned_per_example.csv",
        "gemma": "eval_results/gemma_finetuned_per_example.csv",
        "monarch": "eval_results/monarch_finetuned_per_example.csv",
        "princeton": "eval_results/princeton_finetuned_per_example.csv",
        "seokdong": "eval_results/seokdong_finetuned_per_example.csv",
        "beomi": "eval_results/beomi_finetuned_per_example.csv",
    },
    "base": {
        "bllossom": "eval_results/bllossom_base_per_example.csv",
        "gemma": "eval_results/gemma_base_per_example.csv",
        "monarch": "eval_results/monarch_base_per_example.csv",
        "princeton": "eval_results/princeton_base_per_example.csv",
        "seokdong": "eval_results/seokdong_base_per_example.csv",
        "beomi": "eval_results/beomi_base_per_example.csv",
    }
}

def compute_absolute_scores(csv_path: str) -> dict:
    """
    Compute absolute performance metrics from a per-example evaluation CSV.
    """
    df = pd.read_csv(csv_path)

    # Legal core score: average of three key legal evaluation metrics
    df["legal_core_score"] = (
        df["conclusion_correctness"]
        + df["legal_reasoning"]
        + df["citation_relevance"]
    ) / 3.0

    # Overall score: average of all four
    df["overall_score"] = (
        df["legal_core_score"] * 3
        + df["clarity_and_tone"]
    ) / 4.0

    return {
        "legal_core_mean": df["legal_core_score"].mean(),
        "legal_core_std": df["legal_core_score"].std(),
        "clarity_mean": df["clarity_and_tone"].mean(),
        "clarity_std": df["clarity_and_tone"].std(),
        "overall_mean": df["overall_score"].mean(),
        "overall_std": df["overall_score"].std(),
        "count": len(df),
    }

def run_evaluation(split_name: str, paths: dict, output_file: str):
    rows = []

    for model_name, path in paths.items():
        if not os.path.exists(path):
            print(f"Missing file: {path}")
            continue

        scores = compute_absolute_scores(path)
        rows.append({
            "model": model_name,
            "split": split_name,
            **scores
        })

    summary_df = pd.DataFrame(rows)

    # Sort by LegalCoreScore (primary)
    summary_df = summary_df.sort_values(
        by="legal_core_mean",
        ascending=False
    )

    summary_df.to_csv(output_file, index=False)
    print(summary_df)


if __name__ == "__main__":
    run_evaluation(
        split_name="finetuned",
        paths=RESULT_PATHS["finetuned"],
        output_file="eval_results/absolute_score_summary_finetuned.csv",
    )

    run_evaluation(
        split_name="base",
        paths=RESULT_PATHS["base"],
        output_file="eval_results/absolute_score_summary_base.csv",
    )