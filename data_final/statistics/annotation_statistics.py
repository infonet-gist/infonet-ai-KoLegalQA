import json
import re
import statistics
from collections import defaultdict

INPUT_PATH = "../_kolegalqa_merged_dedup.jsonl"

DATASET_NAME_MAP = {
    "법률상담사례": "KLAC",
    "생활법령백문백답": "EFPL",
    "법령해석사례": "MOLEG",
}

LAW_PATTERN = re.compile(r"<LAW>.*?</LAW>")

stats = defaultdict(lambda: {
    "answer_total": 0,
    "answer_annotated": 0,
    "annotation_total": 0,
    "annotation_num": [],  # for median
})

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        dataset_raw = data.get("dataset", "").strip()
        answer = data.get("answer", "")

        dataset = DATASET_NAME_MAP.get(dataset_raw, dataset_raw)

        matches = LAW_PATTERN.findall(answer)
        annotation_num = len(matches)

        stats[dataset]["answer_total"] += 1
        stats[dataset]["annotation_total"] += annotation_num
        stats[dataset]["annotation_num"].append(annotation_num)

        if annotation_num > 0:
            stats[dataset]["answer_annotated"] += 1

# Print results
print("=" * 80)
print("Statutory Annotation Density Analysis")
print("=" * 80)

for dataset, s in stats.items():
    total = s["answer_total"]
    anned = s["answer_annotated"]
    ann_total = s["annotation_total"]

    cover_ratio = anned / total if total > 0 else 0
    ann_avg = ann_total / total if total > 0 else 0
    ann_med = statistics.median(s["annotation_num"])

    print(f"\n[{dataset}]")
    print(f"Total answers: {total}")
    print(f"Answers with statutory annotations: {anned}")
    print(f"Coverage ratio: {cover_ratio:.4f} ({cover_ratio * 100:.2f}%)")
    print(f"Average annotations per answer: {ann_avg:.4f}")
    print(f"Median annotations per answer: {ann_med:.4f}")