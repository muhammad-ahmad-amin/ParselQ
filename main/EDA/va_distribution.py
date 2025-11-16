import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Paths
TRAIN_PATH = "task-dataset/track_b/subtask_1/eng/eng_environmental_protection_train_task1.jsonl"
SAVE_DIR = "plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load JSONL Data
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

train = load_jsonl(TRAIN_PATH)

# Extract Valence and Arousal
valence = []
arousal = []

for d in train:
    if "Aspect_VA" in d:
        for item in d["Aspect_VA"]:
            if "VA" in item:
                try:
                    v, a = item["VA"].split("#")
                    valence.append(float(v))
                    arousal.append(float(a))
                except Exception as e:
                    print(f"Skipping invalid VA: {item['VA']} ({e})")

print(f"Loaded {len(valence)} valence labels and {len(arousal)} arousal labels.")

# Histogram: Valence
plt.figure(figsize=(7, 4))
plt.hist(valence, bins=20, color="skyblue", edgecolor="black")
plt.title("Valence Distribution")
plt.xlabel("Valence")
plt.ylabel("Frequency")
plt.savefig(f"{SAVE_DIR}/valence_distribution.pdf", bbox_inches="tight")
plt.close()

# Histogram: Arousal
plt.figure(figsize=(7, 4))
plt.hist(arousal, bins=20, color="salmon", edgecolor="black")
plt.title("Arousal Distribution")
plt.xlabel("Arousal")
plt.ylabel("Frequency")
plt.savefig(f"{SAVE_DIR}/arousal_distribution.pdf", bbox_inches="tight")
plt.close()

# Scatter Plot (Valence vs Arousal)
plt.figure(figsize=(6, 6))
plt.scatter(valence, arousal, alpha=0.5, color="purple")
plt.title("Valence vs Arousal")
plt.xlabel("Valence")
plt.ylabel("Arousal")
plt.savefig(f"{SAVE_DIR}/va_scatter.pdf", bbox_inches="tight")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(4, 3))
corr = np.corrcoef(valence, arousal)
sns.heatmap(corr, annot=True, cmap="coolwarm",
            xticklabels=["Valence", "Arousal"],
            yticklabels=["Valence", "Arousal"])
plt.title("Correlation: Valence vs Arousal")
plt.savefig(f"{SAVE_DIR}/va_correlation_heatmap.pdf", bbox_inches="tight")
plt.close()

print("VA distribution plots saved successfully.")