import json
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

TRAIN_PATH = "task-dataset/track_b/subtask_1/eng/eng_environmental_protection_train_task1.jsonl"
PLOT_DIR = "plots"
CSV_DIR = "main/csv"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# Load Data
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

train = load_jsonl(TRAIN_PATH)

# Extract fields
text_lengths = []
valences = []
arousals = []
aspect_va = defaultdict(list)

for item in train:
    text = item.get("Text", "")
    aspects = item.get("Aspect_VA", [])
    
    # Collect text length vs VA (average VA across aspects for that text)
    v_list, a_list = [], []
    for asp in aspects:
        try:
            v, a = asp["VA"].split("#")
            v_list.append(float(v))
            a_list.append(float(a))
            aspect_va[asp["Aspect"]].append((float(v), float(a)))
        except:
            continue
    
    if text and v_list and a_list:
        text_lengths.append(len(text.split()))
        valences.append(np.mean(v_list))
        arousals.append(np.mean(a_list))

# CSV Output: Text-length - VA
with open(f"{CSV_DIR}/length_va_data.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text_length", "valence", "arousal"])
    for L, V, A in zip(text_lengths, valences, arousals):
        writer.writerow([L, V, A])

# Text Length - VA correlation plot
plt.figure(figsize=(8, 5))
plt.scatter(text_lengths, valences, alpha=0.4, label="Valence", color="skyblue")
plt.scatter(text_lengths, arousals, alpha=0.4, label="Arousal", color="salmon")
plt.xlabel("Text Length (#words)")
plt.ylabel("Value")
plt.title("Text Length vs Valence/Arousal")
plt.legend()
plt.savefig(f"{PLOT_DIR}/va_vs_length.pdf", bbox_inches="tight")
plt.close()

# Aspect-level Stats
aspect_names = []
avg_valence = []
avg_arousal = []

for asp, va_list in aspect_va.items():
    if len(va_list) > 0:
        vs = [v for v, a in va_list]
        ars = [a for v, a in va_list]
        aspect_names.append(asp)
        avg_valence.append(np.mean(vs))
        avg_arousal.append(np.mean(ars))

# CSV Output: Aspect averages
with open(f"{CSV_DIR}/aspect_average_va.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["aspect", "avg_valence", "avg_arousal"])
    for asp, av, aa in zip(aspect_names, avg_valence, avg_arousal):
        writer.writerow([asp, av, aa])

# Aspect VA Summary Plot
plt.figure(figsize=(10, 6))
plt.scatter(avg_valence, avg_arousal, alpha=0.6, color="purple")
for i, asp in enumerate(aspect_names):
    plt.text(avg_valence[i] + 0.01, avg_arousal[i] + 0.01, asp, fontsize=8)
plt.xlabel("Average Valence")
plt.ylabel("Average Arousal")
plt.title("Aspect-Level VA Summary")
plt.savefig(f"{PLOT_DIR}/aspect_va_summary.pdf", bbox_inches="tight")
plt.close()

# Extreme VA examples saved to CSV
all_va_entries = []
for item in train:
    for asp in item.get("Aspect_VA", []):
        try:
            v, a = map(float, asp["VA"].split("#"))
            all_va_entries.append({"ID": item["ID"], "Text": item["Text"], 
                                   "Aspect": asp["Aspect"], "Valence": v, "Arousal": a})
        except:
            continue

# Sort by valence and arousal
sorted_by_valence = sorted(all_va_entries, key=lambda x: x["Valence"])
sorted_by_arousal = sorted(all_va_entries, key=lambda x: x["Arousal"])

# Save extreme valence examples
with open(f"{CSV_DIR}/extreme_valence.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "Text", "Aspect", "Valence", "Arousal"])
    # 3 lowest
    for e in sorted_by_valence[:3]:
        writer.writerow([e["ID"], e["Text"], e["Aspect"], e["Valence"], e["Arousal"]])
    # 3 highest
    for e in sorted_by_valence[-3:]:
        writer.writerow([e["ID"], e["Text"], e["Aspect"], e["Valence"], e["Arousal"]])

# Save extreme arousal examples
with open(f"{CSV_DIR}/extreme_arousal.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "Text", "Aspect", "Valence", "Arousal"])
    # 3 lowest
    for e in sorted_by_arousal[:3]:
        writer.writerow([e["ID"], e["Text"], e["Aspect"], e["Valence"], e["Arousal"]])
    # 3 highest
    for e in sorted_by_arousal[-3:]:
        writer.writerow([e["ID"], e["Text"], e["Aspect"], e["Valence"], e["Arousal"]])

print("\nAll CSVs and plots saved successfully.")