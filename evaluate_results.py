import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# =======================
# Paths
# =======================
TRAIN_PATH = r"task-dataset/track_b/subtask_1/eng/eng_environmental_protection_train_task1.jsonl"
DEV_PATH = r"task-dataset/track_b/subtask_1/eng/eng_environmental_protection_dev_task1.jsonl"
PROPOSED_MODEL = r"D:\ParselQ\results\model_proposed.pt"
BASELINE_MODEL = r"D:\ParselQ\models\distilbert_model.pth"  # Example path

PLOTS_DIR = "plots"
RESULTS_DIR = "results"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =======================
# Transformer Block
# =======================
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=512, nhead=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        x2 = F.relu(self.linear1(x))
        x = self.norm2(x + self.linear2(x2))
        return x

# =======================
# Proposed Model
# =======================
class ProposedModel(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=256, num_layers=2, output_dim=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 512, embed_dim))
        self.layers = nn.ModuleList([TransformerLayer(embed_dim=embed_dim) for _ in range(num_layers)])
        self.adapter_down = nn.Linear(embed_dim, 64)
        self.adapter_up = nn.Linear(64, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding[:, :x.size(1), :]
        x = x.transpose(0, 1)  # seq_len x batch_size x embed_dim
        for layer in self.layers:
            x = layer(x)
        x = F.relu(self.adapter_down(x))
        x = self.adapter_up(x)
        x = self.norm(x)
        x = x.mean(dim=0)
        out = self.output_layer(x)
        return out

# =======================
# Wrapper
# =======================
class ModelWrapper:
    def __init__(self, model_class, model_path):
        self.model = model_class()
        state_dict = torch.load(model_path, map_location="cpu")
        if isinstance(state_dict, dict):
            self.model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("Model file is not a state_dict")
        self.model.eval()

    def predict(self, x):
        with torch.no_grad():
            return self.model(x)

# =======================
# Load JSONL
# =======================
def load_dataset(file_path):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

# =======================
# Text + Aspect -> Tensor
# =======================
def text_aspect_to_tensor(text, aspects=None, max_len=50, vocab_size=5000):
    if aspects is None:
        aspects = []
    combined = text + " " + " ".join(aspects)
    vec = [ord(c) % vocab_size for c in combined][:max_len]
    if len(vec) < max_len:
        vec += [0] * (max_len - len(vec))
    return torch.tensor(vec, dtype=torch.long).unsqueeze(0)

# =======================
# Evaluate Model
# =======================
def evaluate_model(wrapper, dataset):
    y_true = []
    y_pred = []
    for item in dataset:
        aspects = item.get("Aspect", [])
        x = text_aspect_to_tensor(item["Text"], aspects)
        pred = wrapper.predict(x)
        predicted_label = torch.argmax(pred, dim=1).item()
        y_pred.append(predicted_label)
        # For demo purposes, we assume label=0 or 1 in 'Label' key
        y_true.append(item.get("Label", 0))  
    return y_true, y_pred

# =======================
# Compute Metrics
# =======================
def compute_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    macro = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )[:3]
    micro = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )[:3]
    weighted = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )[:3]
    return {
        "per_class": {"precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist()},
        "macro": {"precision": macro[0], "recall": macro[1], "f1": macro[2]},
        "micro": {"precision": micro[0], "recall": micro[1], "f1": micro[2]},
        "weighted": {"precision": weighted[0], "recall": weighted[1], "f1": weighted[2]}
    }

# =======================
# Plotting
# =======================
def plot_confusion_matrix(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()

def plot_bar_comparison(baseline_metrics, proposed_metrics, filename):
    labels = ["Precision", "Recall", "F1"]
    baseline_vals = [baseline_metrics["macro"]["precision"], baseline_metrics["macro"]["recall"], baseline_metrics["macro"]["f1"]]
    proposed_vals = [proposed_metrics["macro"]["precision"], proposed_metrics["macro"]["recall"], proposed_metrics["macro"]["f1"]]
    x = range(len(labels))
    plt.figure(figsize=(6,5))
    plt.bar([i-0.15 for i in x], baseline_vals, width=0.3, label="Baseline")
    plt.bar([i+0.15 for i in x], proposed_vals, width=0.3, label="Proposed")
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Baseline vs Proposed (Macro)")
    plt.legend()
    plt.savefig(filename)
    plt.close()

# =======================
# Main
# =======================
if __name__ == "__main__":
    dataset = load_dataset(DEV_PATH)

    # Load models
    baseline_wrapper = ModelWrapper(ProposedModel, BASELINE_MODEL)
    proposed_wrapper = ModelWrapper(ProposedModel, PROPOSED_MODEL)

    # Evaluate
    y_true_b, y_pred_b = evaluate_model(baseline_wrapper, dataset)
    y_true_p, y_pred_p = evaluate_model(proposed_wrapper, dataset)

    # Compute metrics
    baseline_metrics = compute_metrics(y_true_b, y_pred_b)
    proposed_metrics = compute_metrics(y_true_p, y_pred_p)

    # Save metrics
    with open(os.path.join(RESULTS_DIR, "eval_metrics.json"), "w") as f:
        json.dump({"baseline": baseline_metrics, "proposed": proposed_metrics}, f, indent=4)

    # Plot confusion matrix (for proposed)
    plot_confusion_matrix(y_true_p, y_pred_p, os.path.join(PLOTS_DIR, "confusion_matrix.pdf"))

    # Plot baseline vs proposed macro metrics
    plot_bar_comparison(baseline_metrics, proposed_metrics, os.path.join(PLOTS_DIR, "baseline_vs_proposed.pdf"))

    print("Evaluation complete. Metrics and plots saved.")
