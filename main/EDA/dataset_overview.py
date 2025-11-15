import json
import matplotlib.pyplot as plt
from collections import Counter
import os
import csv

# ----------------------------
# CONFIGURATION
# ----------------------------
TRAIN_PATH = "task-dataset/track_b/subtask_1/eng/eng_environmental_protection_train_task1.jsonl"
DEV_PATH   = "task-dataset/track_b/subtask_1/eng/eng_environmental_protection_dev_task1.jsonl"

PLOT_DIR = "plots"
CSV_DIR = "main/csv"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

SAVE_PLOT_PATH = os.path.join(PLOT_DIR, "dataset_overview.pdf")

# ----------------------------
# LOAD DATA
# ----------------------------
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                print(f"Skipping broken line in {path}")
    return data


print("Loading dataset...")
train_data = load_jsonl(TRAIN_PATH)
dev_data = load_jsonl(DEV_PATH)

print(f"Loaded {len(train_data)} train samples, {len(dev_data)} dev samples.\n")

# ----------------------------
# BASIC COUNTS
# ----------------------------
def count_aspects(data):
    total_aspects = 0
    empty_aspects = 0

    for entry in data:
        aspects = entry.get("aspects", [])
        if len(aspects) == 0:
            empty_aspects += 1
        total_aspects += len(aspects)

    avg_aspects = total_aspects / len(data) if len(data) > 0 else 0
    return total_aspects, avg_aspects, empty_aspects


train_aspects, train_avg, train_empty = count_aspects(train_data)
dev_aspects, dev_avg, dev_empty = count_aspects(dev_data)


# SAVE DATASET OVERVIEW CSV
overview_csv = os.path.join(CSV_DIR, "dataset_overview.csv")
with open(overview_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Metric", "Train", "Dev"])
    w.writerow(["Samples", len(train_data), len(dev_data)])
    w.writerow(["Total Aspects", train_aspects, dev_aspects])
    w.writerow(["Avg Aspects", train_avg, dev_avg])
    w.writerow(["Posts with 0 aspects", train_empty, dev_empty])


print("\n✔ Saved:", overview_csv)

# ----------------------------
# EXAMPLE ENTRIES (CSV)
# ----------------------------
example_csv = os.path.join(CSV_DIR, "example_entries.csv")
with open(example_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Index", "Text", "Aspects"])
    for i, ex in enumerate(train_data[:5]):
        w.writerow([i+1, ex.get("text", ""), ";".join(ex.get("aspects", []))])

print("✔ Saved:", example_csv)


# ----------------------------
# VALIDATION CHECKS (CSV)
# ----------------------------
def run_validation(data, name):
    missing_text = sum(1 for d in data if not d.get("text"))
    missing_aspects = sum(1 for d in data if "aspects" not in d)
    wrong_format = sum(
        1 for d in data if "aspects" in d and not isinstance(d["aspects"], list)
    )

    csv_path = os.path.join(CSV_DIR, f"validation_{name.lower()}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Error Type", "Count"])
        w.writerow(["Missing Text", missing_text])
        w.writerow(["Missing 'Aspects' Field", missing_aspects])
        w.writerow(["Non-list 'Aspects' Format", wrong_format])

    print(f"✔ Saved validation CSV for {name}:", csv_path)


run_validation(train_data, "TRAIN")
run_validation(dev_data, "DEV")


# ----------------------------
# ASPECT STATISTICS CSV
# ----------------------------
aspect_counter_train = Counter()
aspect_counter_dev = Counter()

for entry in train_data:
    for a in entry.get("aspects", []):
        aspect_counter_train[a] += 1

for entry in dev_data:
    for a in entry.get("aspects", []):
        aspect_counter_dev[a] += 1

aspect_csv = os.path.join(CSV_DIR, "aspect_stats.csv")
with open(aspect_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Aspect", "Train Count", "Dev Count"])
    for asp in sorted(set(list(aspect_counter_train) + list(aspect_counter_dev))):
        w.writerow([asp, aspect_counter_train.get(asp, 0), aspect_counter_dev.get(asp, 0)])

print("✔ Saved:", aspect_csv)


# ----------------------------
# PLOT (Dataset Size)
# ----------------------------
plt.figure(figsize=(6, 4))
plt.bar(["Train", "Dev"], [len(train_data), len(dev_data)])
plt.ylabel("Number of Samples")
plt.title("Dataset Size: Train vs Dev")

plt.savefig(SAVE_PLOT_PATH, bbox_inches="tight")
plt.close()

print(f"\nSaved plot to {SAVE_PLOT_PATH}")