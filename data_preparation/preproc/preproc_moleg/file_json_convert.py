import json

input_file = 'opinions_parsed.jsonl'
output_file = 'opinions_parsed.json'

objects = []
buffer = ""
brace_count = 0

with open(input_file, 'r', encoding='utf-8') as fin:
    for line in fin:
        line = line.strip()
        if not line:
            continue

        buffer += line
        brace_count += line.count('{')
        brace_count -= line.count('}')

        # a complete JSON object has been assembled
        if brace_count == 0 and buffer:
            try:
                obj = json.loads(buffer)
                objects.append(obj)
            except json.JSONDecodeError as e:
                print("JSON 파싱 실패:", e)
            buffer = ""

with open(output_file, 'w', encoding='utf-8') as fout:
    json.dump(objects, fout, ensure_ascii=False, indent=2)
