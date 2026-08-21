#!/usr/bin/env python3
"""
SakThai CPU training — Qwen2.5-0.5B + LoRA + completion-only loss.
Auto-detects CPU/GPU. Designed to work on this local machine (no CUDA).
"""
import os, torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

HF_USER = "Nanthasit"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LEN = 1024
LIMIT = 20  # small subset for CPU testing; set to None for full data
EPOCHS = 1

is_cpu = not torch.cuda.is_available()
print(f"Device: {'CPU' if is_cpu else 'GPU'}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float32,
)
model.config.use_cache = False

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
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

main = load_dataset(f"{HF_USER}/sakthai-combined-v7", split="train")
if LIMIT:
    main = main.select(range(min(LIMIT, len(main))))
train_data = main.map(to_text, remove_columns=main.column_names)
try:
    supp = load_dataset(f"{HF_USER}/sakthai-irrelevance-supplement", split="train")
    if LIMIT:
        supp = supp.select(range(min(LIMIT, len(supp))))
    supp_text = supp.map(to_text, remove_columns=supp.column_names)
    train_data = concatenate_datasets([train_data, supp_text])
except Exception as e:
    print("irrelevance-supplement skipped:", e)

eval_raw = load_dataset(f"{HF_USER}/sakthai-combined-v7", split="test")
if LIMIT:
    eval_raw = eval_raw.select(range(min(max(5, LIMIT // 4), len(eval_raw))))
eval_data = eval_raw.map(to_text, remove_columns=eval_raw.column_names)
print(f"train={len(train_data)}  eval={len(eval_data)}")

args = SFTConfig(
    output_dir="./sakthai-0.5b-cpu",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
    logging_steps=1, eval_strategy="steps", eval_steps=5,
    save_strategy="no",
    load_best_model_at_end=False,
    fp16=False, bf16=False,
    report_to="none",
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    completion_only_loss=True,
)
trainer = SFTTrainer(
    model=model, processing_class=tokenizer, args=args,
    train_dataset=train_data, eval_dataset=eval_data,
)
trainer.train()
trainer.save_model("./sakthai-0.5b-cpu-final")
tokenizer.save_pretrained("./sakthai-0.5b-cpu-final")
print(f"\nDone. Model saved to ./sakthai-0.5b-cpu-final")
