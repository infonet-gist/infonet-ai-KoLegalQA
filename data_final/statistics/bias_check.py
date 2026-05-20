import pandas as pd
import json
from collections import Counter

sample_df = pd.read_csv("qa_sampled_100.csv")

sample_questions = set(
    sample_df["question"].astype(str).str.strip()
)

full_data = []

with open("../../ko_legal_qa/data_final/_kolegalqa_merged_dedup.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        full_data.append(json.loads(line))


question_to_info = {
    str(item["question"]).strip(): (
        item["source"],
        item["dataset"]
    )
    for item in full_data
}

matched_info = []
not_found = []

for q in sample_questions:
    if q in question_to_info:
        matched_info.append(question_to_info[q])
    else:
        not_found.append(q)

info_counter = Counter(matched_info)

total = len(matched_info)

print("=" * 50)
print(f"Matched samples: {total}")
print(f"Not found samples: {len(not_found)}")
print("=" * 50)

for (source, dataset), count in sorted(
    info_counter.items(),
    key=lambda x: x[1],
    reverse=True
):
    ratio = count / total * 100
    print(f"{source} | {dataset}: {count} ({ratio:.2f}%)")

# print missing questions
if not_found:
    print("\n[Not Found Questions]")
    for q in not_found:
        print(q)