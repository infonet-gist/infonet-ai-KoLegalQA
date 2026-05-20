'''
Manually delete duplicated features.
'''

import json
import pandas as pd

JSONL_PATH = "_kolegalqa_merged.jsonl"
OUTPUT_JSONL = "_kolegalqa_merged_dedup.jsonl"

records = []
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        obj["_row_id"] = i  # trace original row index
        records.append(obj)

df = pd.DataFrame(records)
df = df[df["question"].notnull() & df["answer"].notnull()]
df = df[df["answer"].str.strip().astype(bool)]


df = df.drop_duplicates(subset=["question", "answer"], keep="first")
conflict_groups = (
    df.groupby("question")
      .filter(lambda x: x["answer"].nunique() > 1)
)

print(f"\n* Number of cases with identical questions but different answers: {conflict_groups['question'].nunique()}\n")

rows_to_keep = set()
handled_row_ids = set()

case_id = 1

for question, group in conflict_groups.groupby("question"):
    print("=" * 100)
    print(f"[CASE {case_id}]")
    print("Q:", question)
    print("-" * 100)

    rows = group[["_row_id", "answer", "category"]].reset_index(drop=True)

    for i, row in rows.iterrows():
        print(f"({i+1}) row_id={row['_row_id']} | category={row['category']}")
        ans = row["answer"]
        print(ans[:500] + ("..." if len(ans) > 500 else ""))
        print("-" * 100)

    while True:
        try:
            choice = input(f"Select the answer number to keep (1~{len(rows)}): ").strip()
            choice = int(choice)
            if 1 <= choice <= len(rows):
                break
        except ValueError:
            pass
        print("* Invalid input. Please try again.")

    chosen_row_id = rows.iloc[choice - 1]["_row_id"]
    rows_to_keep.add(chosen_row_id)
    handled_row_ids.update(rows["_row_id"].tolist())

    print(f"* Selected: row_id={chosen_row_id}")
    case_id += 1


all_row_ids = set(df["_row_id"])
non_conflict_ids = all_row_ids - handled_row_ids

final_row_ids = non_conflict_ids | rows_to_keep
final_df = df[df["_row_id"].isin(final_row_ids)]

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for _, row in final_df.drop(columns=["_row_id"]).iterrows():
        f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

print("\n* Final dedup JSONL created successfully")
print(f"  File: {OUTPUT_JSONL}")
print(f"  Total samples: {len(final_df)}")
