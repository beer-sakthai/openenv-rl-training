import math
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_callback import EarlyStoppingCallback

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
OUTPUT_DIR = "./Sakthai-1.5"
MAX_LENGTH = 512
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
EPOCHS = 3
LEARNING_RATE = 5e-5
WARMUP_RATIO = 0.1
USE_LORA = False  # set True for memory-efficient fine-tuning
USE_STREAMING = False
USE_GRADIENT_CHECKPOINTING = True

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(
    DATASET_NAME, DATASET_CONFIG, split="train",
    streaming=USE_STREAMING
)
valid_dataset = load_dataset(
    DATASET_NAME, DATASET_CONFIG, split="validation",
    streaming=USE_STREAMING
)

# Sliding-window tokenization: don't discard overflow
def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        return_overflowing_tokens=True,
        stride=16,
    )
    return outputs

tokenized_dataset = dataset.map(
    tokenize_function, batched=True, remove_columns=dataset.column_names,
)
tokenized_valid = valid_dataset.map(
    tokenize_function, batched=True, remove_columns=valid_dataset.column_names,
)

model_kwargs = dict(
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)
if torch.cuda.is_available():
    model_kwargs["device_map"] = "auto"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

if USE_GRADIENT_CHECKPOINTING:
    model.gradient_checkpointing_enable()

if USE_LORA:
    from peft import LoraConfig, get_peft_model, TaskType
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

def compute_metrics(eval_pred):
    loss = eval_pred.losses if hasattr(eval_pred, 'losses') else eval_pred.loss
    perplexity = math.exp(loss) if loss < 100 else float('inf')
    return {"perplexity": perplexity}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    logging_steps=50,
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=torch.cuda.is_available(),
    fp16=False,
    prediction_loss_only=False,
    report_to="none",
    ddp_find_unused_parameters=False if USE_LORA else None,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_valid,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
