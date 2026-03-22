"""
Fine-tune DistilBERT for query difficulty classification.
Designed for Google Colab free tier (T4 GPU).

Colab setup:
    !pip install transformers torch scikit-learn
    # upload training_data.jsonl
    !python train_classifier.py --data training_data.jsonl --output ./model_output
"""

import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)


# ── Label mapping ───────────────────────────────────────────────
# Dataset has labels: 1 (trivial), 2 (easy), 3 (medium), 5 (hard)
# Map to 0-indexed: 0, 1, 2, 3  (4 classes)

RAW_TO_INDEX = {1: 0, 2: 1, 3: 2, 5: 3}
INDEX_TO_LABEL = {0: "trivial", 1: "easy", 2: "medium", 3: "hard"}
INDEX_TO_SCORE = {0: 1.0, 1: 2.0, 2: 3.0, 3: 5.0}
NUM_LABELS = 4


# ── Dataset ─────────────────────────────────────────────────────

class QueryDifficultyDataset(Dataset):
    def __init__(self, queries, labels, tokenizer, max_len=128):
        self.queries = queries
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.queries[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Data loading ────────────────────────────────────────────────

def load_data(path: str) -> tuple[list[str], list[int]]:
    """Load JSONL dataset. Maps raw labels to 0-indexed."""
    queries, labels = [], []
    skipped = 0
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            raw_label = d["difficulty"]
            if raw_label not in RAW_TO_INDEX:
                skipped += 1
                continue
            queries.append(d["query"])
            labels.append(RAW_TO_INDEX[raw_label])

    print(f"Loaded {len(queries)} queries (skipped {skipped})")
    unique, counts = np.unique(labels, return_counts=True)
    dist = {INDEX_TO_LABEL[u]: int(c) for u, c in zip(unique, counts)}
    print(f"Distribution: {dist}")
    return queries, labels


def train_test_split(queries, labels, test_ratio=0.15):
    """Simple shuffle split."""
    indices = list(range(len(queries)))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - test_ratio))

    train_q = [queries[i] for i in indices[:split]]
    train_l = [labels[i] for i in indices[:split]]
    val_q = [queries[i] for i in indices[split:]]
    val_l = [labels[i] for i in indices[split:]]

    return train_q, train_l, val_q, val_l


# ── Training ────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    queries, labels = load_data(args.data)
    train_q, train_l, val_q, val_l = train_test_split(queries, labels)
    print(f"Train: {len(train_q)}, Val: {len(val_q)}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=NUM_LABELS,
    ).to(device)

    train_ds = QueryDifficultyDataset(train_q, train_l, tokenizer, max_len=args.max_len)
    val_ds = QueryDifficultyDataset(val_q, val_l, tokenizer, max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        val_acc, val_report = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train loss: {avg_loss:.4f}, Train acc: {train_acc:.4f}")
        print(f"  Val acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, tokenizer, args.output)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    # final eval
    print(f"\n── Final Evaluation ──")
    model = DistilBertForSequenceClassification.from_pretrained(args.output).to(device)
    val_acc, val_report = evaluate(model, val_loader, device)
    print(val_report)

    meta = {
        "best_val_acc": round(best_val_acc, 4),
        "num_labels": NUM_LABELS,
        "label_map": INDEX_TO_LABEL,
        "score_map": {str(k): v for k, v in INDEX_TO_SCORE.items()},
        "raw_to_index": {str(k): v for k, v in RAW_TO_INDEX.items()},
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "train_size": len(train_q),
        "val_size": len(val_q),
        "device": str(device),
    }
    with open(Path(args.output) / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nTraining metadata saved to {args.output}/training_meta.json")


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    target_names = [INDEX_TO_LABEL[i] for i in range(NUM_LABELS)]
    report = classification_report(
        all_labels, all_preds,
        target_names=target_names,
        digits=3,
        zero_division=0,
    )
    return acc, report


def save_model(model, tokenizer, output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to training_data.jsonl")
    parser.add_argument("--output", default="./model_output")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()

    train(args)
