# model_baseline_B.py

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from sklearn.metrics import classification_report
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# =====================================================
# Dataset
# =====================================================
class DimStanceDataset(Dataset):
    def __init__(self, file_path):
        self.samples = []

        print(f"[+] Loading dataset: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading JSON Lines"):
                data = json.loads(line)
                text = data.get("Text", "")

                for item in data.get("Aspect_VA", []):
                    aspect = item.get("Aspect", "")
                    va_str = item.get("VA", "5.0#5.0")

                    try:
                        valence, arousal = map(float, va_str.split("#"))
                    except:
                        valence, arousal = 5.0, 5.0

                    combined = f"{text} [SEP] {aspect}"

                    self.samples.append(
                        (combined, np.array([valence, arousal], dtype=np.float32))
                    )

        print(f"[✓] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# =====================================================
# Collate function
# =====================================================
def collate_fn(batch, tokenizer, max_len=128):
    texts = [b[0] for b in batch]
    va_vals = torch.tensor([b[1] for b in batch], dtype=torch.float32)

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"], va_vals


# =====================================================
# BiLSTM Regression Model
# =====================================================
class BiLSTMRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, input_ids, mask):
        x = self.embedding(input_ids)
        lengths = mask.sum(dim=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        out = torch.mean(out, dim=1)
        out = self.dropout(out)
        return self.fc(out)


# =====================================================
# VA → 4 Classes Converter
# =====================================================
def va_to_class(val, aro):
    if val < 5 and aro < 5:
        return 0
    elif val < 5 and aro >= 5:
        return 1
    elif val >= 5 and aro < 5:
        return 2
    else:
        return 3


# =====================================================
# Generate PDF Report
# =====================================================
def generate_pdf_report(y_true, y_pred, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    report_text = classification_report(y_true, y_pred)

    c = canvas.Canvas(save_path, pagesize=letter)
    textobject = c.beginText(40, 750)
    textobject.setFont("Helvetica", 11)

    textobject.textLine("===== BASELINE B REPORT =====")
    textobject.textLine("")
    for line in report_text.split("\n"):
        textobject.textLine(line)

    c.drawText(textobject)
    c.save()
    print(f"[✓] PDF saved at {save_path}")


# =====================================================
# Training
# =====================================================
def train_model(model, train_loader, dev_loader, device, epochs=10, lr=1e-3, patience=3):

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )

    train_losses, dev_losses = [] , []
    best_loss = float("inf")
    no_improve = 0

    print("\n[+] Training started...\n")

    for epoch in range(epochs):
        model.train()        
        total_loss = 0

        prog = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train")
        for ids, mask, va in prog:
            ids, mask, va = ids.to(device), mask.to(device), va.to(device)

            optimizer.zero_grad()
            out = model(ids, mask)
            loss = criterion(out, va)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            prog.set_postfix(loss=loss.item())

        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        # Dev evaluation
        if dev_loader is not None:
            model.eval()
            dev_total = 0
            with torch.no_grad():
                for ids, mask, va in dev_loader:
                    ids, mask, va = ids.to(device), mask.to(device), va.to(device)
                    out = model(ids, mask)
                    dev_total += criterion(out, va).item()

            dev_loss = dev_total / len(dev_loader)
            dev_losses.append(dev_loss)
            scheduler.step(dev_loss)

            print(f"Epoch {epoch+1} | Train={train_loss:.4f} | Dev={dev_loss:.4f}")
        else:
            dev_loss = train_loss
            print(f"Epoch {epoch+1} | Train={train_loss:.4f}")

        # Early stopping
        if dev_loss < best_loss:
            best_loss = dev_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print("[!] Early stopping")
            break

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/bilstm_model.pth")
    print("[✓] Model saved at models/bilstm_model.pth")

    # Save loss curve
    os.makedirs("results", exist_ok=True)
    plt.plot(train_losses, label="Train")
    if dev_loader:
        plt.plot(dev_losses, label="Dev")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("results/bilstm_loss_curve.pdf")
    plt.close()
    print("[✓] Loss curve saved at results/bilstm_loss_curve.pdf")

    # =============================
    # Generate PDF REPORT
    # =============================
    print("[+] Generating Baseline-A style PDF report...")

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for ids, mask, va in train_loader:
            ids, mask, va = ids.to(device), mask.to(device), va.to(device)
            outputs = model(ids, mask)

            for t, p in zip(va.cpu().numpy(), outputs.cpu().numpy()):
                y_true.append(va_to_class(t[0], t[1]))
                y_pred.append(va_to_class(p[0], p[1]))

    generate_pdf_report(y_true, y_pred, "results/baseline_B_report.pdf")

    return model


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    TRAIN_PATH = "task-dataset/track_b/subtask_1/eng/eng_environmental_protection_train_task1.jsonl"
    DEV_PATH   = "task-dataset/track_b/subtask_1/eng/eng_environmental_protection_dev_task1.jsonl"

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = DimStanceDataset(TRAIN_PATH)
    dev_dataset   = DimStanceDataset(DEV_PATH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    dev_loader = (
        DataLoader(dev_dataset, batch_size=16,
                   collate_fn=lambda b: collate_fn(b, tokenizer))
        if len(dev_dataset) > 0 else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMRegressor(vocab_size=tokenizer.vocab_size).to(device)

    train_model(
        model,
        train_loader,
        dev_loader,
        device,
        epochs=10,
        lr=1e-3,
        patience=3
    )
