import math
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import datasets
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from accelerate import Accelerator

MODEL_ID = "mlabonne/NeuralMonarch-7B"
JSON_PATH = "/home/jovyan/_yongtae/ko_legal_qa/data_final/_kolegalqa_merged.jsonl"
OUTPUT_DIR = "./saved"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512
PROMPT_MAX = 256

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
#tokenizer.pad_token = tokenizer.eos_token  # set padding token
if tokenizer.pad_token is None:                      # <<<<
    tokenizer.pad_token = tokenizer.eos_token        # <<<<
tokenizer.padding_side = "right"

bnb4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb4,
    device_map="auto",
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
)

base_model = prepare_model_for_kbit_training(base_model)
model = get_peft_model(base_model, peft_config)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.print_trainable_parameters()

# Tokenization + Masking
def tokenize_with_answer_mask(examples):
    input_ids_list = []
    attention_masks = []
    labels_list = []

    for q, a, c in zip(examples["question"], examples["answer"], examples["category"]):
        prompt = f"""### 질문:\n{q} [분야: {c}]\n\n### 답변:\n"""
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=PROMPT_MAX,
        )["input_ids"]

        remaining = MAX_LENGTH - len(prompt_ids)
        if remaining <= 0:
            prompt_ids = prompt_ids[:MAX_LENGTH]
            answer_ids = []
        else:
            answer_ids = tokenizer(
                a,
                add_special_tokens=False,
                truncation=True,
                max_length=remaining,
            )["input_ids"]

        ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids  # Mask prompt tokens with -100 so loss is only computed on answer tokens

        pad_len = MAX_LENGTH - len(ids)
        if pad_len > 0:
            ids += [tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            attention_mask = [1] * (len(prompt_ids) + len(answer_ids)) + [0] * pad_len
        else:
            attention_mask = [1] * MAX_LENGTH

        input_ids_list.append(ids)
        attention_masks.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_masks,
        "labels": labels_list,
    }

# eval metric
def compute_metrics(eval_pred):
    logits, labels = eval_pred  # logits: (batch, seq, vocab), labels: (batch, seq)
    logits = torch.from_numpy(logits)
    labels = torch.from_numpy(labels)

    # predict next token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    perplexity = math.exp(loss.item())
    return {"perplexity": float(perplexity)}

# Load dataset
raw = datasets.load_dataset("json", data_files=JSON_PATH)["train"]
raw = raw.filter(lambda x: x["answer"] and len(x["answer"].strip()) > 0 and x.get("category") is not None)
df = pd.DataFrame({
    "question": raw["question"],
    "answer": raw["answer"],
    "category": raw["category"]
})
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save test split
test_df.to_csv("qa_oracle_test_split.csv", index=False, encoding="utf-8-sig")


train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(test_df)

train_dataset = train_dataset.map(
    tokenize_with_answer_mask,
    batched=True,
    remove_columns=train_dataset.column_names,
)
eval_dataset = eval_dataset.map(
    tokenize_with_answer_mask,
    batched=True,
    remove_columns=eval_dataset.column_names,
)

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)
eval_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    learning_rate=2e-5,
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    eval_strategy="epoch",
    per_device_eval_batch_size=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    prediction_loss_only=True,
)

# data collator
def data_collator(features):
    # features: list of each sample(dict)
    batch = {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
    }
    return batch

# Accelerator setup <- 사용 안됨
#accelerator = Accelerator()
#optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
#
#model, optimizer, train_dataloader = accelerator.prepare(
#    model,
#    optimizer,
#    torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=training_args.per_device_train_batch_size, collate_fn=data_collator)
#)

# initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# start training
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Training done. Model saved to:", OUTPUT_DIR)