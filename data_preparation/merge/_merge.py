import json

input_files = [
    "easylaw.json",
    "klac_unicodefix.json",
    "moleg_unicodefix.json",
]

output_file = "_kolegalqa_merged.jsonl"

with open(output_file, "w", encoding="utf-8") as out_f:
    for file in input_files:
        with open(file, "r", encoding="utf-8") as in_f:
            data = json.load(in_f)

            for item in data:
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Done")
