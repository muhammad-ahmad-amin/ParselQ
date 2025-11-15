import json
import matplotlib.pyplot as plt
from collections import Counter
import os
import csv

# ----------------------------------------------------
# PATHS
# ----------------------------------------------------
TRAIN_PATH = "task-dataset/track_b/subtask_1/eng/eng_environmental_protection_train_task1.jsonl"
DEV_PATH   = "task-dataset/track_b/subtask_1/eng/eng_environmental_protection_dev_task1.jsonl"

PLOT_DIR = "plots"
CSV_DIR = "main/csv"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)


# ----------------------------------------------------
# LOAD JSONL
# ----------------------------------------------------
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except Exception as e:
                print(f"Skipping broken line in {path}: {e}")
    return data

train_data = load_jsonl(TRAIN_PATH)
dev_data = load_jsonl(DEV_PATH)


# ----------------------------------------------------
# BASIC STATS
# ----------------------------------------------------
def aspect_stats(data):
    total_aspects = sum(len(d.get("aspects", [])) for d in data)
    empty_posts   = sum(1 for d in data if len(d.get("aspects", [])) == 0)
    avg_aspects   = total_aspects / len(data) if len(data) else 0
    return total_aspects, avg_aspects, empty_posts

train_aspects, train_avg, train_empty = aspect_stats(train_data)
dev_aspects, dev_avg, dev_empty = aspect_stats(dev_data)


# ----------------------------------------------------
# CSV 1 — DATASET OVERVIEW
# ----------------------------------------------------
overview_csv = os.path.join(CSV_DIR, "dataset_overview.csv")
with open(overview_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Metric", "Train", "Dev"])
    w.writerow(["Samples", len(train_data), len(dev_data)])
    w.writerow(["Total Aspects", train_aspects, dev_aspects])
    w.writerow(["Avg Aspects", train_avg, dev_avg])
    w.writerow(["Zero-Aspect Posts", train_empty, dev_empty])
print("✔ Saved:", overview_csv)


# ----------------------------------------------------
# ASPECT COUNTERS (TRAIN & DEV)
# ----------------------------------------------------
aspect_counter_train = Counter()
aspect_counter_dev   = Counter()

for d in train_data:
    for a in d.get("aspects", []):
        aspect_counter_train[a] += 1

for d in dev_data:
    for a in d.get("aspects", []):
        aspect_counter_dev[a] += 1


# ----------------------------------------------------
# CSV 2 — ASPECT STATS
# ----------------------------------------------------
aspect_csv = os.path.join(CSV_DIR, "aspect_stats.csv")
with open(aspect_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Aspect", "Train Count", "Dev Count"])
    all_aspects = sorted(set(aspect_counter_train) | set(aspect_counter_dev))
    for a in all_aspects:
        w.writerow([a, aspect_counter_train.get(a, 0), aspect_counter_dev.get(a, 0)])
print("✔ Saved:", aspect_csv)


# ----------------------------------------------------
# CSV 3 — VALIDATION CHECKS
# ----------------------------------------------------
missing_text_train = sum(1 for d in train_data if not d.get("text"))
missing_aspects_train = sum(1 for d in train_data if "aspects" not in d)
wrong_format_train = sum(1 for d in train_data if "aspects" in d and not isinstance(d.get("aspects"), list))

with open(os.path.join(CSV_DIR, "validation.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Error Type", "Train Count"])
    w.writerow(["Missing Text", missing_text_train])
    w.writerow(["Missing 'Aspects' Field", missing_aspects_train])
    w.writerow(["Non-list 'Aspects'", wrong_format_train])
print("✔ Saved:", os.path.join(CSV_DIR, "validation.csv"))


# ----------------------------------------------------
# PLOTTING UTIL
# ----------------------------------------------------
def save_plot(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("✔ Plot saved:", path)


# ----------------------------------------------------
# PLOT 1 — DATASET SIZE
# ----------------------------------------------------
fig = plt.figure(figsize=(7, 5))
plt.bar(["Train", "Dev"], [len(train_data), len(dev_data)])
plt.title("Dataset Size: Train vs Dev")
plt.ylabel("Number of Posts")
save_plot(fig, "dataset_size.pdf")


# ----------------------------------------------------
# PLOT 2 — HISTOGRAM OF ASPECT COUNTS
# ----------------------------------------------------
train_counts = [len(d.get("aspects", [])) for d in train_data]
dev_counts = [len(d.get("aspects", [])) for d in dev_data]

fig = plt.figure(figsize=(7, 5))
plt.hist(train_counts, bins=10, alpha=0.6, label=f"Train (n={len(train_data)})")
plt.hist(dev_counts, bins=10, alpha=0.6, label=f"Dev (n={len(dev_data)})")
plt.xlabel("Number of Aspects")
plt.ylabel("Frequency")
plt.title("Histogram: Aspects Per Post")
plt.legend()
save_plot(fig, "aspect_histogram.pdf")


# ----------------------------------------------------
# PLOT 3 — ZERO VS NON-ZERO ASPECT POSTS (Train & Dev)
# Fixed: produces two pies, handles zero-count edge cases,
# and shows both absolute counts and percentages.
# ----------------------------------------------------
def pie_counts_and_labels(zero_count, total_count):
    nonzero = total_count - zero_count
    sizes = [zero_count, nonzero]
    labels = [f"Zero Aspects ({zero_count})", f"Has Aspects ({nonzero})"]
    return sizes, labels

# Train pie
sizes, labels = pie_counts_and_labels(train_empty, len(train_data))
fig = plt.figure(figsize=(7, 5))
# If both sizes are zero (empty dataset), draw an empty placeholder
if sum(sizes) == 0:
    plt.text(0.5, 0.5, "No data", ha="center", va="center")
else:
    plt.pie(sizes, labels=labels, autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct/100*sum(sizes)))})")
plt.title("Train: Zero vs Non-Zero Aspect Posts")
save_plot(fig, "zero_vs_nonzero_train.pdf")

# Dev pie
sizes, labels = pie_counts_and_labels(dev_empty, len(dev_data))
fig = plt.figure(figsize=(7, 5))
if sum(sizes) == 0:
    plt.text(0.5, 0.5, "No data", ha="center", va="center")
else:
    plt.pie(sizes, labels=labels, autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct/100*sum(sizes)))})")
plt.title("Dev: Zero vs Non-Zero Aspect Posts")
save_plot(fig, "zero_vs_nonzero_dev.pdf")


