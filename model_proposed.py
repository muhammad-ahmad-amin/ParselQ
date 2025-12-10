# model_proposed.py
"""
Proposed Model for SemEval-2026 Dimensional Stance Analysis (Track B)
- Transformer Encoder
- Adapter / Prompt tuning
- LayerNorm + Dropout
- Predicts Valence & Arousal scores (regression)
- Loads real JSONL dataset for training
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Paths to dataset
# -----------------------------
TRAIN_PATH = "task-dataset/track_b/subtask_1/eng/eng_environmental_protection_train_task1.jsonl"
DEV_PATH   = "task-dataset/track_b/subtask_1/eng/eng_environmental_protection_dev_task1.jsonl"

# -----------------------------
# Dataset Class
# -----------------------------
class VADataset(Dataset):
    def __init__(self, path, tokenizer, max_len=50):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj["Text"]
                for aspect_obj in obj["Aspect_VA"]:
                    aspect = aspect_obj["Aspect"]
                    va_str = aspect_obj["VA"]  # "7.00#7.17"
                    valence, arousal = map(float, va_str.split("#"))
                    tokens = tokenizer(text, aspect, max_len=max_len)
                    self.samples.append((tokens, torch.tensor([valence, arousal], dtype=torch.float)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# -----------------------------
# Simple Tokenizer (replace with real tokenizer later)
# -----------------------------
class SimpleTokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size

    def __call__(self, text, aspect, max_len=50):
        # Naive: split words and map to random integers in vocab
        tokens = [hash(word + aspect) % self.vocab_size for word in (text + " " + aspect).split()]
        # Pad / truncate
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        return torch.tensor(tokens, dtype=torch.long)

# -----------------------------
# Model
# -----------------------------
class DimStanceModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, n_heads=8, hidden_dim=512, 
                 n_layers=2, dropout=0.1, max_len=512, adapter_dim=64):
        super(DimStanceModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, 
                                                dim_feedforward=hidden_dim, dropout=dropout, 
                                                activation='relu', batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(embed_dim, 2)

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # [batch, seq, embed]
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x + self.adapter(x)
        x = x.mean(dim=1)  # mean pooling over sequence
        x = self.norm(x)
        x = self.dropout(x)
        va_scores = self.output_layer(x)
        va_scores = torch.clamp(va_scores, 1.0, 9.0)
        return va_scores

# -----------------------------
# Training Function
# -----------------------------
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 5000
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    model = DimStanceModel(vocab_size=vocab_size).to(device)

    train_dataset = VADataset(TRAIN_PATH, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 5

    for epoch in range(n_epochs):
        total_loss = 0
        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "results/model_proposed.pt")
    print("Model weights saved to results/model_proposed.pt")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    train_model()
