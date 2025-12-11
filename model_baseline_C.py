#!/usr/bin/env python3
"""
model_baseline_C.py

BERT-base fine-tuning baseline for stance classification (HuggingFace + PyTorch).

Features:
- Tokenization + attention masks
- CLS pooling (use the pooled output or take hidden_state[0][:,0,:])
- AdamW optimizer + linear warmup scheduler
- Layer-freezing experiment (freeze first N encoder layers or freeze base)
- Save model checkpoints and evaluation metrics
- Print hardware/runtime notes (GPU usage)
"""

import os
import time
import json
import random
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
    AdamW,
)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


# -----------------------
# Utility / Dataset
# -----------------------
class TextDataset(Dataset):
    def _init_(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _len_(self):
        return len(self.texts)

    def _getitem_(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }
        return item


# -----------------------
# Model wrapper
# -----------------------
class BertForStance(nn.Module):
    def _init_(self, model_name, num_labels, dropout_prob=0.1, use_pooler=True):
        super()._init_()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=False)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        self.use_pooler = use_pooler  # if True use pooler_output; else use last_hidden_state[:,0,:]
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # outputs: (last_hidden_state, pooler_output) depending on model
        if self.use_pooler and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output  # (batch, hidden)
        else:
            pooled = outputs.last_hidden_state[:, 0, :]  # CLS token

        x = self.dropout(pooled)
        logits = self.classifier(x)
        return logits


# -----------------------
# Helper functions
# -----------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_encoder_layers(model, freeze_until_layer=None, freeze_all=False):
    """
    Freeze layers of the encoder.
    - freeze_all: freeze entire backbone (bert)
    - freeze_until_layer: integer n means freeze the first n encoder layers (0-based)
      e.g., freeze_until_layer=6 freezes layers 0..5 inclusive.
    """
    if freeze_all:
        for param in model.bert.parameters():
            param.requires_grad = False
        print("[INFO] Entire BERT encoder frozen.")
        return

    if freeze_until_layer is None:
        return

    # BERT-like: encoder layers are in bert.encoder.layer
    # works for models with .bert.encoder.layer (BertModel)
    try:
        enc_layers = model.bert.encoder.layer
        n = len(enc_layers)
        print(f"[INFO] Found {n} encoder layers in model. Freezing first {freeze_until_layer} layers (0-indexed).")
        for i, layer in enumerate(enc_layers):
            if i < freeze_until_layer:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True
    except Exception:
        print("[WARN] Could not find encoder layers to freeze. Model may differ; skipping layer freezing.")


def save_metrics(metrics: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def plot_confusion_matrix(cm, labels, out_path):
    if not HAS_MATPLOTLIB:
        print("[WARN] matplotlib not available, skipping confusion matrix plot.")
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -----------------------
# Training & Evaluation
# -----------------------
def train_epoch(model, dataloader, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * input_ids.size(0)
        pbar.set_postfix({"loss": loss.item()})
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def eval_model(model, dataloader, device, label_names=None):
    model.eval()
    y_true = []
    y_pred = []
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=label_names, digits=4, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    return acc, report, cm, y_true, y_pred


# -----------------------
# Main runner
# -----------------------
def main(args):
    set_seed(args.seed)

    # -----------------------
    # Hardware / runtime notes
    # -----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU count: {torch.cuda.device_count()}, name: {torch.cuda.get_device_name(0)}")
        try:
            import torch.cuda as cuda
            print(f"[INFO] GPU memory allocated: {cuda.memory_allocated(0) / (1024**3):.2f} GB")
        except Exception:
            pass

    # -----------------------
    # Load CSV dataset (expects 'text' and 'label' columns)
    # -----------------------
    if not os.path.exists(args.data_csv):
        raise FileNotFoundError(f"Data CSV not found: {args.data_csv}")
    df = pd.read_csv(args.data_csv)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns (label integers or strings).")

    # Convert string labels to integer ids if needed
    label_vals = list(sorted(df["label"].unique()))
    # build mapping if labels are strings
    if not np.issubdtype(df["label"].dtype, np.number):
        label2id = {lab: i for i, lab in enumerate(label_vals)}
        id2label = {i: lab for lab, i in label2id.items()}
        df["label"] = df["label"].map(label2id)
    else:
        id2label = {i: str(i) for i in label_vals}
        label2id = {str(i): i for i in label_vals}

    num_labels = len(set(df["label"].tolist()))
    label_names = [id2label[i] for i in range(num_labels)]
    print(f"[INFO] Number of labels: {num_labels}, labels: {label_names}")

    # Split train/dev/test (simple split)
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(df, test_size=args.test_size + args.val_size, random_state=args.seed, stratify=df["label"])
    val_size_rel = args.val_size / (args.test_size + args.val_size)
    val_df, test_df = train_test_split(temp_df, test_size=val_size_rel, random_state=args.seed, stratify=temp_df["label"])

    print(f"[INFO] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # -----------------------
    # Tokenizer & Datasets
    # -----------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = TextDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_len=args.max_len)
    val_dataset = TextDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_len=args.max_len)
    test_dataset = TextDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_len=args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # -----------------------
    # Model init
    # -----------------------
    model = BertForStance(args.model_name, num_labels=num_labels, dropout_prob=args.dropout, use_pooler=not args.no_pooler)
    model.to(device)

    # Freeze layers if requested
    if args.freeze_all:
        freeze_encoder_layers(model, freeze_all=True)
    elif args.freeze_until_layer is not None:
        freeze_encoder_layers(model, freeze_until_layer=args.freeze_until_layer)

    # Show number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable parameters: {trainable_params:,} / {total_params:,}")

    # -----------------------
    # Optimizer & Scheduler
    # -----------------------
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

    t_total = len(train_loader) * args.epochs
    warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    # -----------------------
    # Training loop
    # -----------------------
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_acc": []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        train_loss = train_epoch(model, train_loader, optimizer, device, scheduler=scheduler)
        val_acc, val_report, val_cm, _, _ = eval_model(model, val_loader, device, label_names=label_names)

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)

        print(f"[INFO] Epoch {epoch} train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(out_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[INFO] Saved best model to {ckpt_path}")

    elapsed = time.time() - start_time
    print(f"[INFO] Training completed in {elapsed/60:.2f} minutes. Best val acc: {best_val_acc:.4f}")

    # Save history
    with open(os.path.join(out_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # -----------------------
    # Final evaluation on test set (using best model)
    # -----------------------
    # load best model
    best_path = os.path.join(out_dir, "best_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"[INFO] Loaded best model from {best_path}")
    else:
        print("[WARN] Best model checkpoint not found; using current model weights.")

    test_acc, test_report, test_cm, y_true, y_pred = eval_model(model, test_loader, device, label_names=label_names)
    print("\n===== TEST SET RESULTS =====")
    print(f"Accuracy: {test_acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=label_names, digits=4))

    # Save metrics
    metrics = {
        "test_accuracy": float(test_acc),
        "test_report": test_report,
        "train_history": history,
        "label_names": label_names,
    }
    save_metrics(metrics, out_dir)
    print(f"[INFO] Metrics saved to {out_dir}/metrics.json")

    # Save confusion matrix plot
    cm_path = os.path.join(out_dir, "confusion_matrix.pdf")
    try:
        plot_confusion_matrix(test_cm, label_names, cm_path)
        print(f"[INFO] Confusion matrix plot saved to {cm_path}")
    except Exception as e:
        print(f"[WARN] Could not save confusion matrix plot: {e}")

    # Save model (final)
    final_path = os.path.join(out_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"[INFO] Final model saved to {final_path}")

    # Save classification report text
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, target_names=label_names, digits=4))

    # Hardware/runtime summary
    runtime_summary = {
        "device": str(device),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "best_val_acc": float(best_val_acc),
        "training_minutes": elapsed / 60.0,
    }
    with open(os.path.join(out_dir, "runtime_summary.json"), "w", encoding="utf-8") as f:
        json.dump(runtime_summary, f, indent=2)

    print("[INFO] Done.")


# -----------------------
# CLI
# -----------------------
if _name_ == "_main_":
    parser = argparse.ArgumentParser(description="Fine-tune BERT-base for Stance Classification")

    parser.add_argument("--data_csv", type=str, default="dataset.csv", help="Path to CSV with columns 'text','label'")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="HuggingFace model name")
    parser.add_argument("--output_dir", type=str, default="results/baseline_C", help="Where to save models and metrics")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_proportion", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_pooler", action="store_true", help="Do not use model pooler output; use CLS token instead")
    parser.add_argument("--freeze_all", action="store_true", help="Freeze all BERT encoder parameters")
    parser.add_argument("--freeze_until_layer", type=int, default=None, help="Freeze first N encoder layers (0-indexed)")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Absolute warmup steps (overrides proportion if set)")

    args = parser.parse_args()
    # If warmup_steps provided override warmup_proportion
    if args.warmup_steps is not None:
        args.warmup_proportion = 0.0  # we'll compute warmup_steps explicitly
    main(args)