import os
import pandas as pd

RESULTS = {
    "base":         "eval_results/gpt_oss_20b_legal_base_per_example.csv",
    "finetuned":    "eval_results/gpt_oss_20b_legal_finetuned_per_example.csv",
}

rows = []

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

for model_name, path in RESULTS.items():
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        continue

    scores = compute_absolute_scores(path)
    rows.append({
        "model": model_name,
        **scores
    })

summary_df = pd.DataFrame(rows)

# Sort by LegalCoreScore (primary)
summary_df = summary_df.sort_values(
    by="legal_core_mean",
    ascending=False
)

summary_df.to_csv(
    "eval_results/absolute_score_summary.csv",
    index=False
)

print(summary_df)
