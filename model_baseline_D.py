# model_baseline_D.py
"""
DistilBERT baseline for DimStance (DimASR) regression task.
Saves model to models/distilbert_model.pth and tokenizer to models/distilbert_tokenizer/
Saves results plots and runtime summary to results/
"""

import os
import json
import time
import math
import psutil
from datetime import datetime
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW


from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
from tqdm import tqdm

# --------- Configurable hyperparameters ----------
MODEL_NAME = "distilbert-base-uncased"
TRAIN_PATH = "task-dataset/track_b/subtask_1/eng/eng_environmental_protection_train_task1.jsonl"
DEV_PATH   = "task-dataset/track_b/subtask_1/eng/eng_environmental_protection_dev_task1.jsonl"

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 6
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 0
OUTPUT_DIR = "models"
RESULTS_DIR = "results"
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --------- Utilities ----------
def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_memory_usage_mb():
    # Return process memory usage in MB (RSS)
    try:
        p = psutil.Process()
        return p.memory_info().rss / (1024 * 1024)
    except Exception:
        return None

# --------- Dataset ----------
class DimStanceDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=MAX_LEN):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                text = data.get('Text', "")
                for item in data.get('Aspect_VA', []):
                    aspect = item.get('Aspect', "")
                    va_str = item.get('VA', None)
                    if va_str is None:
                        # default neutral
                        valence, arousal = 5.0, 5.0
                    else:
                        try:
                            valence, arousal = map(float, va_str.split("#"))
                        except:
                            valence, arousal = 5.0, 5.0
                    combined = f"{text} [SEP] {aspect}"
                    self.samples.append({
                        "text": combined,
                        "valence": float(valence),
                        "arousal": float(arousal)
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        encoding = self.tokenizer(
            s["text"],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        targets = torch.tensor([s["valence"], s["arousal"]], dtype=torch.float32)
        return input_ids, attention_mask, targets

# --------- Model (DistilBERT + small regression head) ----------
class DistilBertRegressor(nn.Module):
    def __init__(self, model_name=MODEL_NAME, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)  # DistilBERT
        hidden_size = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, 2)  # valence, arousal

        # initialize head
        nn.init.xavier_uniform_(self.regressor.weight)
        if self.regressor.bias is not None:
            nn.init.constant_(self.regressor.bias, 0.0)

    def forward(self, input_ids, attention_mask):
        # DistilBERT returns last_hidden_state (batch, seq_len, hidden)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (B, L, H)
        # Use [CLS]-like pooling: use first token or mean pooling
        # DistilBERT doesn't have a pooler; we can use mean pooling over attention mask
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        summed = (last_hidden * mask).sum(1)  # (B, H)
        lengths = mask.sum(1)  # (B, 1)
        pooled = summed / lengths.clamp(min=1e-9)
        pooled = self.drop(pooled)
        out = self.regressor(pooled)  # (B, 2)
        return out

# --------- Helpers for training / evaluation ----------
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    losses = []
    mae_list = []
    start = time.time()
    for input_ids, attention_mask, targets in tqdm(dataloader, desc="Train batch", leave=False):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        outputs = model(input_ids, attention_mask)  # (B,2)
        loss = nn.MSELoss()(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            preds = outputs.detach().cpu().numpy()
            targs = targets.detach().cpu().numpy()
            mae = mean_absolute_error(targs, preds)

        losses.append(loss.item())
        mae_list.append(mae)

    elapsed = time.time() - start
    return np.mean(losses), np.mean(mae_list), elapsed

def evaluate(model, dataloader, device):
    model.eval()
    preds_all = []
    targs_all = []
    with torch.no_grad():
        for input_ids, attention_mask, targets in tqdm(dataloader, desc="Eval batch", leave=False):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask).cpu().numpy()
            preds_all.append(outputs)
            targs_all.append(targets.numpy())
    if len(preds_all) == 0:
        return None
    preds = np.vstack(preds_all)
    targs = np.vstack(targs_all)
    mse_val = mean_squared_error(targs, preds)
    mae_val = mean_absolute_error(targs, preds)
    return {"preds": preds, "targs": targs, "mse": mse_val, "mae": mae_val}

# discretize continuous VA to 3 classes for confusion matrix
def discretize_va(vals, neg_thresh=4.5, pos_thresh=5.5):
    # vals: array shape (N,) or (N,2)
    flat = np.array(vals)
    if flat.ndim == 2:
        # produce per-dimension discretized ints
        classes = []
        for col in range(flat.shape[1]):
            arr = flat[:, col]
            cl = np.zeros_like(arr, dtype=int)
            cl[arr <= neg_thresh] = 0  # negative
            cl[(arr > neg_thresh) & (arr < pos_thresh)] = 1  # neutral
            cl[arr >= pos_thresh] = 2  # positive
            classes.append(cl)
        # returns list [valence_classes, arousal_classes]
        return classes
    else:
        arr = flat
        cl = np.zeros_like(arr, dtype=int)
        cl[arr <= neg_thresh] = 0
        cl[(arr > neg_thresh) & (arr < pos_thresh)] = 1
        cl[arr >= pos_thresh] = 2
        return cl

def plot_learning_curves(train_losses, val_losses, train_mae, val_mae, outpath):
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure()
    plt.plot(epochs, train_losses, label="Train MSE")
    if val_losses:
        plt.plot(epochs, val_losses, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("DistilBERT MSE Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

    plt.figure()
    plt.plot(epochs, train_mae, label="Train MAE")
    if val_mae:
        plt.plot(epochs, val_mae, label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("DistilBERT MAE Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath.replace(".pdf", "_mae.pdf"))
    plt.close()

def plot_confusion(cm, labels, title, outpath):
    # cm is square matrix
    plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    # annotate
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# --------- Main training pipeline ----------
def main():
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load datasets
    train_dataset = DimStanceDataset(TRAIN_PATH, tokenizer)
    dev_dataset = DimStanceDataset(DEV_PATH, tokenizer)

    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty. Check TRAIN_PATH and file format.")
    print(f"Train samples: {len(train_dataset)} | Dev samples: {len(dev_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE) if len(dev_dataset) > 0 else None

    model = DistilBertRegressor(MODEL_NAME)
    model.to(device)

    # Prepare optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

    # Training loop
    train_losses = []
    val_losses = []
    train_mae = []
    val_mae = []
    epoch_times = []
    memory_peak = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        epoch_start = time.time()
        mem_before = get_memory_usage_mb()
        tr_loss, tr_mae, epoch_time = train_epoch(model, train_loader, optimizer, scheduler, device)
        mem_after = get_memory_usage_mb()
        train_losses.append(tr_loss)
        train_mae.append(tr_mae)
        epoch_times.append(epoch_time)
        memory_peak.append(mem_after if mem_after is not None else 0)

        print(f"Train MSE: {tr_loss:.4f} | Train MAE: {tr_mae:.4f} | epoch time {epoch_time:.1f}s | mem (MB): {mem_after}")

        # Validation
        if dev_loader is not None:
            eval_res = evaluate(model, dev_loader, device)
            if eval_res is not None:
                val_losses.append(eval_res["mse"])
                val_mae.append(eval_res["mae"])
                print(f"Val MSE: {eval_res['mse']:.4f} | Val MAE: {eval_res['mae']:.4f}")
            else:
                val_losses.append(None)
                val_mae.append(None)
        else:
            val_losses.append(None)
            val_mae.append(None)

    # Save model weights and tokenizer
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, "distilbert_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save tokenizer
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "distilbert_tokenizer"))

    # Final evaluation & metrics
    if dev_loader is not None:
        final_eval = evaluate(model, dev_loader, device)
        if final_eval is not None:
            preds = final_eval["preds"]
            targs = final_eval["targs"]
            final_mse = final_eval["mse"]
            final_mae = final_eval["mae"]
            print(f"\nFinal Dev MSE: {final_mse:.4f} | Final Dev MAE: {final_mae:.4f}")
        else:
            preds, targs = None, None
    else:
        preds, targs = None, None

    # Plot learning curves
    plot_learning_curves(train_losses, val_losses if any(v is not None for v in val_losses) else None,
                         train_mae, val_mae if any(v is not None for v in val_mae) else None,
                         os.path.join(RESULTS_DIR, "distilbert_learning_curve.pdf"))
    print("Learning curves saved to results/")

    # Confusion matrices: discretize predictions and targets into 3 classes and save
    if preds is not None and targs is not None:
        # per-dimension confusion matrices
        valence_pred_cls, arousal_pred_cls = discretize_va(preds)
        valence_targ_cls, arousal_targ_cls = discretize_va(targs)

        labels = ["neg", "neu", "pos"]
        cm_val = confusion_matrix(valence_targ_cls, valence_pred_cls, labels=[0,1,2])
        cm_ar = confusion_matrix(arousal_targ_cls, arousal_pred_cls, labels=[0,1,2])

        plot_confusion(cm_val, labels, "Valence Confusion Matrix", os.path.join(RESULTS_DIR, "confusion_valence.pdf"))
        plot_confusion(cm_ar, labels, "Arousal Confusion Matrix", os.path.join(RESULTS_DIR, "confusion_arousal.pdf"))
        print("Confusion matrices saved to results/")

    # Save runtime summary
    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device": str(device),
        "train_samples": len(train_dataset),
        "dev_samples": len(dev_dataset),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "epoch_times_sec": epoch_times,
        "memory_mb_samples": memory_peak,
        "final_dev_mse": float(final_mse) if (preds is not None and targs is not None) else None,
        "final_dev_mae": float(final_mae) if (preds is not None and targs is not None) else None
    }
    with open(os.path.join(RESULTS_DIR, "runtime_summary.json"), "w", encoding="utf-8") as wf:
        json.dump(summary, wf, indent=2)
    print("Runtime summary saved to results/runtime_summary.json")

    print("\nDone.")

if __name__ == "__main__":
    main()
