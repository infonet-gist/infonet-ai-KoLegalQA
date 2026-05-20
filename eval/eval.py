import os
import json
import time
import logging
import pandas as pd
import numpy as np
import httpx
from openai import OpenAI
from typing import Dict
from json import JSONDecodeError

# Settings
def load_api_key(path: str = "api_key.txt") -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"API key file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

http_client = httpx.Client(
    timeout=httpx.Timeout(
        connect=10.0,
        read=30.0,
        write=10.0,
        pool=10.0
    )
)

# OpenAI client
api_key = load_api_key("api_key.txt")   # do not upload #
client = OpenAI(api_key=api_key, http_client=http_client, max_retries=0)

OUTPUT_DIR = "eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PAIRS: Dict[str, Dict[str, str]] = {
    "bllossom": {
        "base": "outputs/bllossom/base_output.csv",
        "finetuned": "outputs/bllossom/trained_output.csv",
    },
    "gemma": {
        "base": "outputs/gemma/base_output.csv",
        "finetuned": "outputs/gemma/trained_output.csv",
    },
    "monarch": {
        "base": "outputs/monarch/base_output.csv",
        "finetuned": "outputs/monarch/trained_output.csv",
    },
    "princeton": {
        "base": "outputs/princeton/base_output.csv",
        "finetuned": "outputs/princeton/trained_output.csv",
    },
    "seokdong": {
        "base": "outputs/seokdong/base_output.csv",
        "finetuned": "outputs/seokdong/trained_output.csv",
    },
    "beomi": {
        "base": "outputs/beomi/base_output.csv",
        "finetuned": "outputs/beomi/trained_output.csv",
    },
}

SCORE_COLUMNS = [
    "conclusion_correctness",
    "legal_reasoning",
    "citation_relevance",
    "clarity_and_tone",
]


# Evaluation prompt
EVAL_PROMPT_TEMPLATE = """
당신은 한국 법률 질문–답변을 평가하는 엄격하고 객관적인 평가자입니다.

아래의 [질문]과 [답변]을 읽고, 각 평가 기준에 대해 1점부터 5점까지 점수를 부여하십시오.

점수 기준:
- 1점: 매우 그렇지 않다
- 3점: 부분적으로 그렇다
- 5점: 매우 그렇다
(2점과 4점은 인접한 기준 사이의 중간 수준을 의미함)

[평가 기준]
1. 결론의 정당성:
    질문에서 요구한 법적 쟁점에 대해 올바른 결론을 제시하고 있는가
2. 법적 근거의 적절성:
    제시된 법적 근거가 결론을 정당화하는 데 충분한가
3. 인용의 정확성:
    인용된 법조문 또는 판례는 질문의 쟁점과 직접적으로 관련되는가
4. 서술 명료성과 답변 태도:
    답변의 문장이 명확하고 이해하기 쉬운가

[질문]
{question}

[답변]
{prediction}

반드시 아래 JSON 형식으로만 응답하십시오.
추가 설명은 절대 포함하지 마십시오.

{{
  "conclusion_correctness": <1-5>,
  "legal_reasoning": <1-5>,
  "citation_relevance": <1-5>,
  "clarity_and_tone": <1-5>
}}
"""

# G-eval
def gpt4o_score(question, answer, max_retries=3, sleep_sec=2):
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 점수 부여 외의 어떠한 설명도 출력하지 않는 평가자입니다."
                    },
                    {
                        "role": "user",
                        "content": EVAL_PROMPT_TEMPLATE.format(
                            question=question,
                            prediction=answer
                        )
                    }
                ],
                temperature=0,
            )

            content = response.choices[0].message.content.strip()

            # remove codeblocks
            if content.startswith("```"):
                content = content.strip("```").strip()
                if content.startswith("json"):
                    content = content[4:].strip()

            return json.loads(content)

        except (JSONDecodeError, KeyError, IndexError) as e:
            last_exception = e
            logging.warning(
                f"[Retry {attempt}/{max_retries}] JSON parse failed. "
                f"Question hash={hash(question)}"
            )
            time.sleep(sleep_sec)

        except Exception as e:
            last_exception = e
            logging.error(
                f"[Retry {attempt}/{max_retries}] Unexpected error: {e}"
            )
            time.sleep(sleep_sec)

    # retry failed
    logging.error(
        f"[SKIP] Failed after {max_retries} retries. "
        f"Question hash={hash(question)} | Last error={last_exception}"
    )
    return None


# Single scv evaluation
def evaluate_single_model(csv_path: str, output_prefix: str, pair_name: str, model_role: str, failed_samples: list) -> str:
    df = pd.read_csv(csv_path)
    df = df[df["question"].notnull() & df["prediction"].notnull()]

    rows = []
    for _, row in df.iterrows():
        scores = gpt4o_score(row["question"], row["prediction"])

        if scores is None:
            failed_samples.append({
                "pair": pair_name,
                "model_role": model_role,
                "question": row["question"],
                "prediction": row["prediction"],
            })
            continue

        rows.append({
            "question": row["question"],
            **scores
        })

    result_df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_per_example.csv")
    result_df.to_csv(out_path, index=False)
    return out_path

# Pair-wise analysis
def analyze_pair(pair_name: str, base_csv: str, ft_csv: str) -> pd.DataFrame:
    df_base = pd.read_csv(base_csv)
    df_ft = pd.read_csv(ft_csv)

    df = df_base.merge(
        df_ft,
        on="question",
        suffixes=("_base", "_ft")
    )

    for col in SCORE_COLUMNS:
        df[f"delta_{col}"] = df[f"{col}_ft"] - df[f"{col}_base"]

    out_path = os.path.join(OUTPUT_DIR, f"{pair_name}_paired_results.csv")
    df.to_csv(out_path, index=False)

    return df

# Run the full experiment
if __name__ == "__main__":

    failed_samples = []
    all_pair_deltas = []

    for pair_name, paths in CSV_PAIRS.items():
        print(f"\n=== Processing {pair_name} ===")

        base_out = evaluate_single_model(
            paths["base"],
            f"{pair_name}_base",
            pair_name=pair_name,
            model_role="base",
            failed_samples=failed_samples
        )

        ft_out = evaluate_single_model(
            paths["finetuned"],
            f"{pair_name}_finetuned",
            pair_name=pair_name,
            model_role="finetuned",
            failed_samples=failed_samples
        )

        paired_df = analyze_pair(pair_name, base_out, ft_out)

        pair_summary = {"pair": pair_name}
        for col in SCORE_COLUMNS:
            pair_summary[f"delta_{col}_mean"] = paired_df[f"delta_{col}"].mean()

        all_pair_deltas.append(pair_summary)

# Summary
    summary_df = pd.DataFrame(all_pair_deltas)
    summary_path = os.path.join(OUTPUT_DIR, "all_pairs_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n=== All Pairs Summary ===")
    print(summary_df)
    print(f"\nSaved to: {summary_path}")

    if failed_samples:
        failed_df = pd.DataFrame(failed_samples)
        failed_path = os.path.join(OUTPUT_DIR, "failed_judge_samples.csv")
        failed_df.to_csv(failed_path, index=False)
        print(f"\nSaved failed samples to: {failed_path}")
