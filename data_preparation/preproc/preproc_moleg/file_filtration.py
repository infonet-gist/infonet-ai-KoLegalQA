import json

input_file = 'opinions_parsed.json'
output_file = 'opinions_extracted.json'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

results = []

for obj in data:
    question = obj.get("question", "")

    answer = obj.get("answer", "")
    reason = obj.get("reason", "")
    combined_answer = answer + "\n" + reason if reason else answer

    new_obj = {
        "question": question,
        "answer": combined_answer,
        "source": "법제처",
        "type": "현행법령",
        "dataset": "법령해석사례",
        "original_category": "민원",
        "category": "-"
    }

    results.append(new_obj)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
