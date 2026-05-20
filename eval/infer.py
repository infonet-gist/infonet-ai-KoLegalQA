import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# Paths
BASE_DIR = ".."
TRAIN_DIR = os.path.join(BASE_DIR, "train")
EVAL_DIR = os.path.join(BASE_DIR, "eval")
OUTPUT_BASE = os.path.join(EVAL_DIR, "outputs")
TEST_PATH = os.path.join(BASE_DIR, "data_final/held_out_split.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = [
    {
        "model_name": "bllossom",
        "base_model": "MLP-KTLim/llama-3-Korean-Bllossom-8B",
    },
    {
        "model_name": "gemma",
        "base_model": "recoilme/recoilme-gemma-2-9B-v0.4",
    },
    {
        "model_name": "monarch",
        "base_model": "mlabonne/NeuralMonarch-7B",
    },
    {
        "model_name": "princeton",
        "base_model": "princeton-nlp/Llama-3-8B-ProLong-512k-Base",
    },
    {
        "model_name": "seokdong",
        "base_model": "SEOKDONG/llama3.1_korean_v1.1_sft_by_aidx",
    },
    {
        "model_name": "beomi",
        "base_model": "beomi/Llama-3-Open-Ko-8B",
    },
]

def run_inference(model, tokenizer, questions, categories):
    predictions = []
    batch_size=8
    
    for i in tqdm(range(0, len(questions), batch_size), total=(len(questions)+batch_size-1)//batch_size, mininterval=10, leave=False):
        batch_qs = questions[i:i+batch_size]
        batch_cs = categories[i:i+batch_size]
        batch_prompt = [f"### 질문:\n{q} [분야: {c}]\n\n### 답변:\n" for q, c in zip(batch_qs, batch_cs)]

        inputs = tokenizer(
            batch_prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_lens = inputs["input_ids"].shape[1]
        for j in range(len(outputs)):
            generated = outputs[j][input_lens:]
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            predictions.append(decoded.strip())

    return predictions

# Load data
df = pd.read_csv(TEST_PATH)
questions = list(df["question"])
categories = list(df["category"])
answers = list(df["answer"])

bnb4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Inference
for cfg in MODELS:
    model_name = cfg["model_name"]
    base_model_name = cfg["base_model"]
    print(f"\nRunning inference: {model_name}, Base model: {base_model_name}")
    
    out_dir = os.path.join(OUTPUT_BASE, model_name)
    os.makedirs(out_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base Model
    print(" ▶ Base model inference")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb4, device_map="auto", )
    base_model.eval()
    base_pred = run_inference(base_model, tokenizer, questions, categories)
    
    base_df = pd.DataFrame({
        "question": questions,
        "category": categories,
        "ground_truth": answers,
        "prediction": base_pred,
    })
    csv_path = os.path.join(out_dir, "base_output.csv")
    base_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {csv_path}")

    # Trained Model
    print(" ▶ Trained model inference")
    adapter_dir = os.path.join(TRAIN_DIR, model_name, "saved")
    trained_pred = None
    trained_model = PeftModel.from_pretrained(base_model, adapter_dir)
    
    del base_model
    torch.cuda.empty_cache()
    
    trained_model.eval()
    trained_pred = run_inference(trained_model, tokenizer, questions, categories)
    
    trained_df = pd.DataFrame({
        "question": questions,
        "category": categories,
        "ground_truth": answers,
        "prediction": trained_pred,
    })
    trained_df.to_csv(os.path.join(out_dir, "trained_output.csv"), index=False, encoding="utf-8-sig")
    print(f"Saved: {out_dir}")

    del trained_model
    torch.cuda.empty_cache()