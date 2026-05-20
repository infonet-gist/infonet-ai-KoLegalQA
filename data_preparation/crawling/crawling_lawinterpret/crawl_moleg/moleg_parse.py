import re
import json
from pathlib import Path
from tqdm import tqdm

INPUT = Path("opinions.jsonl")
OUTPUT = Path("opinions_parsed.jsonl")

# Common header
HEADER_PATTERN = re.compile(
    r"법령해석 사례.*?인쇄", re.DOTALL
)

# Common footer
FOOTER_PATTERN = re.compile(
    r"※ 법제처 법령해석의 효력 등에 대한 안내.*?$", re.DOTALL
)

# Main extraction pattern
EXTRACT_PATTERN = re.compile(
    r"(?P<title>.+?)"
    r"안건번호\s*(?P<case_id>[0-9]{2}-[0-9]{4})\s*"
    r"회신일자\s*(?P<reply_date>[0-9]{4}-[0-9]{2}-[0-9]{2})\s*"
    r"1\.\s*질의요지\s*(?P<question>.+?)"
    r"2\.\s*(?:회답|회신)\s*(?P<answer>.+?)"
    r"3\.\s*이유\s*(?P<reason>.+)",
    re.DOTALL
)

def clean_text(t: str) -> str:
    t = HEADER_PATTERN.sub("", t)
    t = FOOTER_PATTERN.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def parse_content(raw: str) -> dict:
    raw = clean_text(raw)
    m = EXTRACT_PATTERN.search(raw)
    if not m:
        return {}  # Preserce original content if parsing fails

    gd = m.groupdict()
    # Normalize whitespace for extracted fiedls
    for k in gd.keys():
        gd[k] = gd[k].strip()
    return gd

def main():
    if not INPUT.exists():
        print("Source file not found:", INPUT)
        return

    if OUTPUT.exists():
        print("Removing existing output file", OUTPUT)
        OUTPUT.unlink()

    total = sum(1 for _ in open(INPUT, encoding="utf-8"))

    with open(INPUT, encoding="utf-8") as fin, \
         open(OUTPUT, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, total=total, desc="Parsing"):
            data = json.loads(line)

            if "content" not in data:
                continue

            parsed = parse_content(data["content"])

            if parsed:
                data.update(parsed)
            else:
                data["parsing_error"] = True

            fout.write(json.dumps(data, ensure_ascii=False, indent=4) + "\n")

    print(f"\nResults saved:\n- {OUTPUT}\n")

if __name__ == "__main__":
    main()
