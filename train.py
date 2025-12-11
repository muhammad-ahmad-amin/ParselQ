import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

# ============================================================
# Dataset Loader
# ============================================================

class DimStanceDataset(Dataset):
    def __init__(self, raw_items, tokenizer, max_len=256):
        self.samples = []
        self.tok = tokenizer
        self.max_len = max_len

        for data in raw_items:
            if "Text" not in data or "Aspect_VA" not in data:
                continue

            for item in data["Aspect_VA"]:
                try:
                    aspect = item["Aspect"]
                    valence, arousal = map(float, item["VA"].split("#"))
                except:
                    continue

                text = data["Text"]
                
                encoded = self.tok(
                    f"aspect: {aspect} text: {text}",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt"
                )

                self.samples.append({
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                    "labels": torch.tensor([valence, arousal], dtype=torch.float)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# Model
# ============================================================

class RegressionModel(torch.nn.Module):
    def __init__(self, base_model="bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        self.regressor = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        out = self.regressor(pooled)
        return out


# ============================================================
# JSONL Reader
# ============================================================

def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except:
                pass
    return items


# ============================================================
# Training Loop
# ============================================================

def train_model(model, train_loader, dev_loader, device, epochs=5, lr=1e-4):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # REMOVE verbose=True (caused your error)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    model.to(device)

    for epoch in range(epochs):
        print(f"\n====== Epoch {epoch+1}/{epochs} ======")

        # ------------------- Train --------------------
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            preds = model(input_ids, mask)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        print(f"Train Loss: {avg_train:.4f}")

        # ------------------- Validation --------------------
        if dev_loader and len(dev_loader) > 0:
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch in tqdm(dev_loader, desc="Validating"):
                    input_ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    preds = model(input_ids, mask)
                    loss = loss_fn(preds, labels)
                    val_loss += loss.item()

            avg_val = val_loss / len(dev_loader)
            print(f"Val Loss: {avg_val:.4f}")

            scheduler.step(avg_val)

        else:
            print("[!] No dev set — skipping validation.")

    return model


# ============================================================
# Main Script
# ============================================================

def main():
    TRAIN_PATH = "task-dataset/track_b/subtask_1/eng/eng_environmental_protection_train_task1.jsonl"
    DEV_PATH = "task-dataset/track_b/subtask_1/eng/eng_environmental_protection_dev_task1.jsonl"

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    print("[+] Loading training set:", TRAIN_PATH)
    train_items = read_jsonl(TRAIN_PATH)
    print(f"[✓] Loaded {len(train_items)} raw samples")

    print("[+] Loading dev set:", DEV_PATH)
    dev_items = read_jsonl(DEV_PATH)
    print(f"[✓] Loaded {len(dev_items)} raw samples")

    # If dev is empty → use 10% split
    if len(dev_items) == 0:
        print("[!] Dev set empty — splitting training data into 90/10")
        train_items, dev_items = train_test_split(train_items, test_size=0.1, random_state=42)

    # Build datasets
    train_dataset = DimStanceDataset(train_items, tokenizer)
    dev_dataset = DimStanceDataset(dev_items, tokenizer)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False)

    # Model
    model = RegressionModel()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trained = train_model(model, train_loader, dev_loader, device, epochs=5)

    # Save to models/
    os.makedirs("models", exist_ok=True)
    save_path = "models/best_model.pt"

    torch.save(trained.state_dict(), save_path)
    print(f"\n[✓] Model saved at: {save_path}")


if __name__ == "__main__":
    main()
