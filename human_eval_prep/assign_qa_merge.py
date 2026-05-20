import os
import pandas as pd

INPUT_DIR = "evaluator_assignment"  # Note: run assign_qa.py first
OUTPUT_FILE = "human_eval_assigned.csv"

csv_files = [
    f for f in os.listdir(INPUT_DIR)
    if f.endswith(".csv")
]

if not csv_files:
    raise ValueError("No CSV files found in directory.")

print(f"Found {len(csv_files)} files.")

# Merge and sort rows
all_dfs = []

for file in csv_files:
    path = os.path.join(INPUT_DIR, file)
    df = pd.read_csv(path)

    all_dfs.append(df)

master_df = pd.concat(all_dfs, ignore_index=True)

master_df = master_df.sort_values(
    by=["evaluator_id", "model_id", "batch_id", "item_id"]
)

master_df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved merged file to: {OUTPUT_FILE}")
print(f"Total rows: {len(master_df)}")
print(f"Unique evaluators: {master_df['evaluator_id'].nunique()}")
print(f"Unique item_id x model_id pairs: {master_df[['item_id','model_id']].drop_duplicates().shape[0]}")
