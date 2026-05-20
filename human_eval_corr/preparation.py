import pandas as pd
import glob
import os

# =====================================
# Human rating preprocessing
# =====================================

human_path = "source/response_table_0219_0600.csv"
human_df = pd.read_csv(human_path)

# Compute aggregated scores at (item × model) level
human_agg = (
    human_df
    .groupby(["item_id", "model"])
    .agg({
        "score_conclusion": "mean",
        "score_reasoning": "mean",
        "score_relevance": "mean",
        "score_clarity": "mean"
    })
    .reset_index()
)

human_agg["human_legal_core"] = (
    human_agg["score_conclusion"]
    + human_agg["score_reasoning"]
    + human_agg["score_relevance"]
) / 3

human_agg["human_overall"] = (
    human_agg["score_conclusion"]
    + human_agg["score_reasoning"]
    + human_agg["score_relevance"]
    + human_agg["score_clarity"]
) / 4

# Rename detailed score columns
human_agg = human_agg.rename(columns={
    "score_conclusion": "human_conclusion",
    "score_reasoning": "human_reasoning",
    "score_relevance": "human_relevance",
    "score_clarity": "human_clarity"
})

print("Human aggregated shape:", human_agg.shape)


# =====================================
# Aggregate g-eval files
# =====================================

geval_files = glob.glob("source/g_eval_results/*_per_example.csv")
geval_list = []

for path in geval_files:
    df = pd.read_csv(path)

    # Extract model name from filename
    # exampel: beomi_base_per_example.csv
    filename = os.path.basename(path)
    model_name = filename.replace("_per_example.csv", "")
    df["model_name_raw"] = model_name
    geval_list.append(df)

geval_all = pd.concat(geval_list, ignore_index=True)

print("All g-eval shape:", geval_all.shape)


# =====================================
# Filter only 100 human-evaluated items
# =====================================

sample_path = "source/qa_sampled_100.csv"
sample_df = pd.read_csv(sample_path)

geval_filtered = geval_all.merge(
    sample_df[["id", "question"]],
    on="question",
    how="inner"
)

# id column -> item_id
geval_filtered = geval_filtered.rename(columns={"id": "item_id"})

print("Filtered g-eval shape:", geval_filtered.shape)


# =====================================
# Map model aliases (a~f) to actual model names
# =====================================
MODEL_MAP = {
    "beomi_base": "a",
    "beomi_finetuned": "b",
    "princeton_base": "c",
    "princeton_finetuned": "d",
    "seokdong_base": "e",
    "seokdong_finetuned": "f"
}

geval_filtered["model"] = geval_filtered["model_name_raw"].map(MODEL_MAP)

# =====================================
# Compute g-eval scores
# =====================================

geval_filtered = geval_filtered.rename(columns={
    "conclusion_correctness": "g_conclusion",
    "legal_reasoning": "g_reasoning",
    "citation_relevance": "g_relevance",
    "clarity_and_tone": "g_clarity"
})

geval_filtered["g_legal_core"] = (
    geval_filtered["g_conclusion"]
    + geval_filtered["g_reasoning"]
    + geval_filtered["g_relevance"]
) / 3

geval_filtered["g_overall"] = (
    geval_filtered["g_conclusion"]
    + geval_filtered["g_reasoning"]
    + geval_filtered["g_relevance"]
    + geval_filtered["g_clarity"]
) / 4

geval_final = geval_filtered[[
    "item_id",
    "model",
    "g_conclusion",
    "g_reasoning",
    "g_relevance",
    "g_clarity",
    "g_legal_core",
    "g_overall"
]]

print("Final g-eval shape:", geval_final.shape)


# =====================================
# Merge human ratings and g-eval scores
# =====================================

analysis_df = pd.merge(
    human_agg,
    geval_final,
    on=["item_id", "model"],
    how="inner"
)

print("Final merged shape:", analysis_df.shape)

analysis_df.to_csv("analysis_ready.csv", index=False)

print("Saved as analysis_ready.csv")
