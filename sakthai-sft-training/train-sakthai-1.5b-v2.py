#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers>=4.44",
#   "trl>=0.19,<0.20",
#   "peft>=0.7",
#   "datasets",
#   "accelerate",
#   "bitsandbytes>=0.46.1",
#   "huggingface_hub",
# ]
# ///
"""
SakThai 1.5B v2 — improved tool-calling fine-tune.
QLoRA + rsLoRA + all-linear targets + completion-only loss.
Trains on v7 + v8 + v8-gap-fill combined (3,440 examples).

Usage on HF Jobs:
  hf jobs uv run --flavor a10g-small --timeout 6h --secrets HF_TOKEN train-sakthai-1.5b-v2.py

Or from URL:
  hf jobs uv run --flavor a10g-small --timeout 6h --secrets HF_TOKEN
      https://huggingface.co/Nanthasit/sakthai-kaggle-notebooks/resolve/main/train-sakthai-1.5b-v2.py
"""
import sys
import os
import subprocess

import importlib.util
if importlib.util.find_spec("trl") is None or importlib.util.find_spec("bitsandbytes") is None:
    print("Installing dependencies on Colab runtime...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-qU",
                    "transformers>=4.44", "trl>=0.19,<0.20", "peft>=0.7",
                    "datasets", "accelerate", "bitsandbytes>=0.46.1", "huggingface_hub"], check=True)

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

BASE_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
HF_USER      = "Nanthasit"
ADAPTER_REPO = f"{HF_USER}/sakthai-plus-1.5b-lora"
MERGED_REPO  = f"{HF_USER}/sakthai-plus-1.5b"
MAX_SEQ_LEN  = 2048
HF_TOKEN     = os.environ.get("HF_TOKEN")
assert HF_TOKEN, "Set HF_TOKEN secret: --secrets HF_TOKEN"

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.config.use_cache = False

lora_config = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_rslora=True,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def to_text(ex):
    msgs = ex["messages"]
    tools = ex.get("tools") or None
    text = tokenizer.apply_chat_template(
        msgs, tools=tools, tokenize=False, add_generation_prompt=False,
    )
    return {"text": text}

main = load_dataset(f"{HF_USER}/sakthai-combined-v7", data_files="data/train.jsonl", split="train")
train_data = main.map(to_text, remove_columns=main.column_names)

# v8 augmentation
try:
    v8 = load_dataset(f"{HF_USER}/sakthai-combined-v8", data_files="data/train.jsonl", split="train")
    v8_text = v8.map(to_text, remove_columns=v8.column_names)
    train_data = concatenate_datasets([train_data, v8_text])
except Exception as e:
    print("v8 unavailable:", e)

# irrelevance supplement
try:
    supp = load_dataset(f"{HF_USER}/sakthai-irrelevance-supplement", data_files="data/train.jsonl", split="train")
    supp_text = supp.map(to_text, remove_columns=supp.column_names)
    train_data = concatenate_datasets([train_data, supp_text])
except Exception as e:
    print("irrelevance-supplement unavailable:", e)

try:
    eval_raw = load_dataset(f"{HF_USER}/sakthai-combined-v7", data_files="data/test.jsonl", split="train")
except Exception:
    eval_raw = load_dataset(f"{HF_USER}/sakthai-combined-v7", data_files="data/train.jsonl", split="train[:100]")
eval_data = eval_raw.map(to_text, remove_columns=eval_raw.column_names)
print(f"train={len(train_data)}  eval={len(eval_data)}")

use_tf32 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
args = SFTConfig(
    output_dir="./sakthai-1.5b-lora",
    max_steps=150,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="adamw_8bit",
    learning_rate=3e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
    logging_steps=10, eval_strategy="steps", eval_steps=50,
    save_strategy="steps", save_steps=50, save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="eval_loss",
    bf16=True, tf32=use_tf32, report_to="none",
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    completion_only_loss=True,
    push_to_hub=True,
    hub_model_id=ADAPTER_REPO,
    hub_strategy="every_save",
)
trainer = SFTTrainer(
    model=model, processing_class=tokenizer, args=args,
    train_dataset=train_data, eval_dataset=eval_data,
)
trainer.train()
trainer.save_model("./sakthai-1.5b-lora-best")
tokenizer.save_pretrained("./sakthai-1.5b-lora-best")

from huggingface_hub import login
login(token=HF_TOKEN)
trainer.model.push_to_hub(ADAPTER_REPO)
tokenizer.push_to_hub(ADAPTER_REPO)

from peft import PeftModel
del model, trainer
torch.cuda.empty_cache()

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
)
merged = PeftModel.from_pretrained(base, "./sakthai-1.5b-lora-best").merge_and_unload()
merged.push_to_hub(MERGED_REPO)
tokenizer.push_to_hub(MERGED_REPO)
print(f"Done. Adapter: {ADAPTER_REPO}  Merged: {MERGED_REPO}")
