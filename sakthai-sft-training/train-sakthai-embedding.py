#!/usr/bin/env python3
# /// script
# dependencies = [
#   "torch",
#   "sentence-transformers>=3.0.0",
#   "datasets",
#   "accelerate",
#   "huggingface_hub",
# ]
# ///
"""Multilingual embedding fine-tuning and semantic tool-retrieval evaluator.

Integrates the workflows from the Google Drive Colab notebooks:
  - `sakthai-embedding-multilingual.ipynb` (`Nanthasit/sakthai-embedding-multilingual`)
  - `fine-tuning-embeddinggemma-with-sentence-transformers.ipynb`
    (`SentenceTransformerTrainer`, `SentenceTransformerTrainingArguments`,
    `MultipleNegativesRankingLoss` on `(anchor, positive, negative)` triplets)
  - `Embeddings.ipynb` (top-k semantic retrieval & passage reranking)

Purpose in the SakThai ecosystem (`docs/MODELS.md`):
  1. Fine-tunes and evaluates `Nanthasit/sakthai-embedding-multilingual` (or
     `google/embeddinggemma-300m`) on bilingual (Thai + English) tool-routing
     and agentic memory triplets.
  2. Evaluates Top-1 / Top-3 semantic tool retrieval recall — pre-filtering a
     large tool catalog down to the top-k candidate tools before prompting
     `sakthai-context-0.5b-tools` or `sakthai-context-7b-tools`.

Usage:
    # Evaluate existing checkpoint only (no training)
    python3 sakthai-sft-training/train-sakthai-embedding.py --eval-only

    # Fine-tune and optionally push to Hub
    hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \\
      -e EMBED_BASE=Nanthasit/sakthai-embedding-multilingual \\
      -e EMBED_PUSH_TO=Nanthasit/sakthai-embedding-multilingual \\
      sakthai-sft-training/train-sakthai-embedding.py
"""

import argparse
import os
from typing import Dict, List, Tuple

DEFAULT_BASE_MODEL = "Nanthasit/sakthai-embedding-multilingual"

# Bilingual (Thai & English) triplets: (anchor_query, positive_tool_doc, negative_tool_doc)
TOOL_RETRIEVAL_TRIPLETS: List[Tuple[str, str, str]] = [
    (
        "ช่วยเช็คสภาพอากาศที่กรุงเทพพรุ่งนี้ให้หน่อย",
        "get_weather(city, date): Retrieve current weather forecast, temperature, and rain probability for a city.",
        "get_stock_price(ticker): Fetch real-time equity price and daily volume from the exchange.",
    ),
    (
        "What is the current weather and humidity in Chiang Mai?",
        "get_weather(city, date): Retrieve current weather forecast, temperature, and rain probability for a city.",
        "send_email(recipient, subject, body): Send an electronic mail message to a contact.",
    ),
    (
        "คำนวณภาษีมูลค่าเพิ่ม 7% ของยอดสั่งซื้อ 1,250 บาท",
        "compute_vat(amount_thb, rate_percent): Calculate Thai Value Added Tax (VAT) and total invoice amount in THB.",
        "translate_text(text, target_lang): Translate a passage between Thai and English.",
    ),
    (
        "Look up the payment status and total for order ORD-202",
        "lookup_order(order_id): Query customer name, order status, and THB amount from the order database.",
        "run_command(command): Execute a bash command inside the isolated sandbox workspace.",
    ),
    (
        "ค้นหาข้อมูลคำสั่งซื้อหมายเลข ORD-101 ว่าชำระเงินหรือยัง",
        "lookup_order(order_id): Query customer name, order status, and THB amount from the order database.",
        "click(bid): Click a DOM element by its BrowserGym identifier on the current web page.",
    ),
    (
        "Run pytest in the workspace to see why the unit test fails",
        "run_command(command): Execute a shell command in the sandboxed terminal and return stdout/stderr/exit_code.",
        "lookup_order(order_id): Query customer name, order status, and THB amount from the order database.",
    ),
    (
        "เขียนไฟล์ answer.txt ที่มีค่า 42 ลงในโฟลเดอร์ทำงาน",
        "write_file(path, content): Create or overwrite a text file inside the sandboxed workspace.",
        "get_weather(city, date): Retrieve current weather forecast, temperature, and rain probability for a city.",
    ),
    (
        "Click the submit button on the login form in the browser",
        "run_action(action): Execute a BrowserGym web navigation action such as click('a12') or fill('a5', 'text').",
        "compute_vat(amount_thb, rate_percent): Calculate Thai Value Added Tax (VAT) and total invoice amount in THB.",
    ),
]

TOOL_CATALOG_DOCS: Dict[str, str] = {
    "get_weather": "get_weather(city, date): Retrieve current weather forecast, temperature, and rain probability for a city.",
    "compute_vat": "compute_vat(amount_thb, rate_percent): Calculate Thai Value Added Tax (VAT) and total invoice amount in THB.",
    "lookup_order": "lookup_order(order_id): Query customer name, order status, and THB amount from the order database.",
    "run_command": "run_command(command): Execute a shell command in the sandboxed terminal and return stdout/stderr/exit_code.",
    "write_file": "write_file(path, content): Create or overwrite a text file inside the sandboxed workspace.",
    "run_action": "run_action(action): Execute a BrowserGym web navigation action such as click('a12') or fill('a5', 'text').",
    "get_stock_price": "get_stock_price(ticker): Fetch real-time equity price and daily volume from the exchange.",
    "send_email": "send_email(recipient, subject, body): Send an electronic mail message to a contact.",
}

EVAL_QUERIES: List[Tuple[str, str]] = [
    ("วันนี้อากาศที่ภูเก็ตเป็นอย่างไร ฝนตกไหม", "get_weather"),
    ("Calculate 7% VAT on 4,800 THB", "compute_vat"),
    ("ช่วยเช็คสถานะออเดอร์ ORD-404 ให้ที", "lookup_order"),
    ("Execute `ls -la` in the terminal sandbox", "run_command"),
    ("Save the upper-case words into upper.txt", "write_file"),
    ("Click element b14 on the web page", "run_action"),
]


def build_triplet_dataset():
    """Build a Hugging Face Dataset with (anchor, positive, negative) columns."""
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "anchor": [t[0] for t in TOOL_RETRIEVAL_TRIPLETS],
            "positive": [t[1] for t in TOOL_RETRIEVAL_TRIPLETS],
            "negative": [t[2] for t in TOOL_RETRIEVAL_TRIPLETS],
        }
    )


def evaluate_tool_retrieval(model) -> Dict[str, float]:
    """Compute Top-1 accuracy, Top-3 recall, and mean positive similarity on EVAL_QUERIES."""
    import torch

    tool_names = list(TOOL_CATALOG_DOCS.keys())
    tool_docs = [TOOL_CATALOG_DOCS[k] for k in tool_names]
    queries = [q for q, _ in EVAL_QUERIES]
    gold_tools = [g for _, g in EVAL_QUERIES]

    q_emb = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True)
    d_emb = model.encode(tool_docs, convert_to_tensor=True, normalize_embeddings=True)
    sims = torch.matmul(q_emb, d_emb.T)

    top1_hits = 0
    top3_hits = 0
    pos_sims: List[float] = []

    for i, gold in enumerate(gold_tools):
        gold_idx = tool_names.index(gold)
        row_scores = sims[i]
        pos_sims.append(float(row_scores[gold_idx].item()))
        ranked_indices = torch.argsort(row_scores, descending=True).tolist()
        if ranked_indices[0] == gold_idx:
            top1_hits += 1
        if gold_idx in ranked_indices[:3]:
            top3_hits += 1

    n = len(EVAL_QUERIES)
    return {
        "top1_accuracy": 100.0 * top1_hits / n,
        "top3_recall": 100.0 * top3_hits / n,
        "mean_positive_cosine": sum(pos_sims) / n,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune and evaluate SakThai multilingual embeddings for tool retrieval."
    )
    parser.add_argument(
        "--base-model",
        default=os.environ.get("EMBED_BASE", DEFAULT_BASE_MODEL),
        help=f"Base SentenceTransformer model ID (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("EMBED_OUT", "./sakthai-embedding-ft"),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.environ.get("EMBED_EPOCHS", "4")),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("EMBED_BATCH", "4")),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=float(os.environ.get("EMBED_LR", "2e-5")),
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=os.environ.get("EMBED_EVAL_ONLY", "0") == "1",
        help="Run semantic tool retrieval evaluation without training.",
    )
    parser.add_argument(
        "--push-to-hub",
        default=os.environ.get("EMBED_PUSH_TO", ""),
        help="Optional Hugging Face Hub repo ID to push fine-tuned embedding model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model {args.base_model} on {device} ...")
    model = SentenceTransformer(args.base_model, device=device)

    before_metrics = evaluate_tool_retrieval(model)
    print("\n--- Semantic Tool Retrieval (Before Fine-Tuning) ---")
    print(f"  Top-1 Accuracy:       {before_metrics['top1_accuracy']:.1f}%")
    print(f"  Top-3 Recall:         {before_metrics['top3_recall']:.1f}%")
    print(f"  Mean Positive Cosine: {before_metrics['mean_positive_cosine']:.4f}")

    if args.eval_only:
        return 0

    train_dataset = build_triplet_dataset()
    loss = MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )
    print(f"\nFine-tuning on {len(train_dataset)} bilingual triplets for {args.epochs} epochs ...")
    trainer.train()

    after_metrics = evaluate_tool_retrieval(model)
    print("\n--- Semantic Tool Retrieval (After Fine-Tuning) ---")
    print(f"  Top-1 Accuracy:       {after_metrics['top1_accuracy']:.1f}%")
    print(f"  Top-3 Recall:         {after_metrics['top3_recall']:.1f}%")
    print(f"  Mean Positive Cosine: {after_metrics['mean_positive_cosine']:.4f}")

    model.save_pretrained(args.output_dir)
    print(f"Saved checkpoint to {args.output_dir}")

    if args.push_to_hub:
        print(f"Pushing fine-tuned embedding model to {args.push_to_hub} ...")
        model.push_to_hub(args.push_to_hub)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
