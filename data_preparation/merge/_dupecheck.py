import json
import pandas as pd

JSONL_PATH = "_kolegalqa_merged.jsonl"

records = []
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        records.append({
            "question": obj.get("question"),
            "answer": obj.get("answer"),
            "category": obj.get("category"),
        })

df = pd.DataFrame(records)

# Remove invalid rows using the same criteria as the training pipeline
df = df[df["question"].notnull() & df["answer"].notnull()]
df = df[df["answer"].str.strip().astype(bool)]

# Count duplicated questions
num_dup_question = df["question"].duplicated().sum()
print(f"Number of duplicated questions: {num_dup_question}")

# Retrieve rows with duplicated questions
dup_questions = df[df["question"].duplicated(keep=False)] \
    .sort_values("question")

print("Sample duplicated questions (top 10):")
print(dup_questions[["question", "answer", "category"]].head(10))

# Count duplicated question-answer pairs
num_dup_qa = df.duplicated(subset=["question", "answer"]).sum()
print(f"Number of duplicated QA pairs: {num_dup_qa}")