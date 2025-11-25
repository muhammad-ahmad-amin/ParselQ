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

# =====================
# Dataset class
# =====================
class DimStanceDataset(Dataset):
    def __init__(self, file_path):
        self.samples = []

        print(f"[+] Loading dataset: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading JSON Lines"):
                data = json.loads(line)
                text = data.get('Text', "")

                for item in data.get('Aspect_VA', []):
                    aspect = item.get('Aspect', "")
                    va_str = item.get('VA', "5.0#5.0")
                    try:
                        valence, arousal = map(float, va_str.split("#"))
                    except:
                        valence, arousal = 5.0, 5.0

                    combined_text = f"{text} [SEP] {aspect}"
                    self.samples.append((combined_text, np.array([valence, arousal], dtype=np.float32)))

        print(f"[✓] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# =====================
# Collate function for fast batching
# =====================
def collate_fn(batch, tokenizer, max_len=128):
    texts = [b[0] for b in batch]
    va_values = torch.tensor([b[1] for b in batch], dtype=torch.float32)

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    return enc['input_ids'], enc['attention_mask'], va_values


# =====================
# Bi-LSTM Model
# =====================
class BiLSTMRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, dropout=0.3):
        super(BiLSTMRegressor, self).__init__()
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

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, c_n) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out = torch.mean(out, dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


# =====================
# Training function
# =====================
def train_model(model, train_loader, dev_loader, device, epochs=10, lr=1e-3, patience=3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5  # Removed verbose parameter
    )

    best_loss = float("inf")
    no_improve_count = 0
    train_losses, dev_losses = [], []

    print("\n[+] Starting training...\n")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        prog = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training")
        for input_ids, att_mask, va in prog:
            input_ids, att_mask, va = (
                input_ids.to(device),
                att_mask.to(device),
                va.to(device),
            )

            optimizer.zero_grad()
            outputs = model(input_ids, att_mask)
            loss = criterion(outputs, va)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping

            optimizer.step()
            epoch_loss += loss.item()
            prog.set_postfix(loss=loss.item())

        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        if dev_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                prog_val = tqdm(dev_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation")
                for input_ids, att_mask, va in prog_val:
                    input_ids, att_mask, va = (
                        input_ids.to(device),
                        att_mask.to(device),
                        va.to(device),
                    )
                    outputs = model(input_ids, att_mask)
                    loss = criterion(outputs, va)
                    val_loss += loss.item()
                    prog_val.set_postfix(loss=loss.item())

            dev_loss = val_loss / len(dev_loader)
            dev_losses.append(dev_loss)
            scheduler.step(dev_loss)

            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Dev: {dev_loss:.4f}")
            
            # Print learning rate for debugging
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.2e}")
        else:
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f}")
            dev_loss = train_loss

        # Early stopping
        if dev_loss < best_loss:
            best_loss = dev_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print("\n[!] Early stopping triggered")
            break

    # =====================
    # Save trained model
    # =====================
    os.makedirs("models", exist_ok=True)
    model_path = "models/bilstm_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n[✓] Model saved → {model_path}")

    # =====================
    # Save loss plot
    # =====================
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    if dev_loader is not None:
        plt.plot(dev_losses, label='Dev Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Bi-LSTM Loss Curves")
    plt.legend()
    plt.savefig("results/bilstm_loss_curve.pdf")
    print("[✓] Loss curve saved to results/bilstm_loss_curve.pdf")
    plt.close()

    return model


# =====================
# Main execution
# =====================
if __name__ == "__main__":
    TRAIN_PATH = "task-dataset/track_b/subtask_1/eng/eng_environmental_protection_train_task1.jsonl"
    DEV_PATH   = "task-dataset/track_b/subtask_1/eng/eng_environmental_protection_dev_task1.jsonl"

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = DimStanceDataset(TRAIN_PATH)
    dev_dataset = DimStanceDataset(DEV_PATH)

    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty! Check dataset format.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    if len(dev_dataset) == 0:
        print("[!] Dev dataset empty! Validation disabled.")
        dev_loader = None
    else:
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=16,
            collate_fn=lambda b: collate_fn(b, tokenizer)
        )

    vocab_size = tokenizer.vocab_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMRegressor(vocab_size=vocab_size).to(device)

    trained_model = train_model(
        model,
        train_loader,
        dev_loader,
        device,
        epochs=10,
        lr=1e-3,
        patience=3
    )