import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer
import numpy as np
import csv

nltk.download('stopwords')

# Config
TRAIN_PATH = "task-dataset/track_b/subtask_1/eng/eng_environmental_protection_train_task1.jsonl"
DEV_PATH   = "task-dataset/track_b/subtask_1/eng/eng_environmental_protection_dev_task1.jsonl"

PLOT_DIR   = "plots"
CSV_DIR    = "main/csv"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

stop_words = set(stopwords.words("english"))
tokenizer = RegexpTokenizer(r'\w+')

# Load data
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

train = load_jsonl(TRAIN_PATH)
dev = load_jsonl(DEV_PATH)

all_data = train + dev
texts = [d["Text"] for d in all_data if "Text" in d]
all_aspects = [aspect for d in all_data if "Aspect" in d for aspect in d["Aspect"]]

print(f"Loaded {len(train)} train samples, {len(dev)} dev samples.")

# Clean and tokenize
def clean_and_tokenize(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

tokenized_texts = [clean_and_tokenize(t) for t in texts]
token_lengths = [len(toks) for toks in tokenized_texts]

# Save token length csv
token_csv_path = os.path.join(CSV_DIR, "token_length_stats.csv")
with open(token_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Mean", np.mean(token_lengths)])
    writer.writerow(["Median", np.median(token_lengths)])
    writer.writerow(["Min", np.min(token_lengths)])
    writer.writerow(["Max", np.max(token_lengths)])

# Plot token length distribution
plt.figure(figsize=(8, 5))
plt.hist(token_lengths, bins=40)
plt.xlabel("Token Count")
plt.ylabel("Frequency")
plt.title("Token Length Distribution")
plt.savefig(os.path.join(PLOT_DIR, "token_length_distribution.pdf"))
plt.close()

# Top tokens
token_counter = Counter()
for toks in tokenized_texts:
    token_counter.update(toks)

top_tokens = token_counter.most_common(30)
tokens, freqs = zip(*top_tokens)

# CSV for top tokens
with open(os.path.join(CSV_DIR, "top_tokens.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Token", "Frequency"])
    writer.writerows(top_tokens)

plt.figure(figsize=(10, 6))
plt.bar(tokens, freqs)
plt.xticks(rotation=75, ha="right")
plt.title("Top 30 Most Frequent Tokens")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "top_tokens.pdf"))
plt.close()

# Top aspects
if all_aspects:
    aspect_counter = Counter(all_aspects)
    top_aspects = aspect_counter.most_common(20)

    # CSV for aspects
    with open(os.path.join(CSV_DIR, "top_aspects.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Aspect", "Count"])
        writer.writerows(top_aspects)

    aspects, counts = zip(*top_aspects)

    plt.figure(figsize=(10, 6))
    plt.bar(aspects, counts)
    plt.xticks(rotation=75, ha="right")
    plt.title("Most Frequent Aspect Terms")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "aspect_frequency.pdf"))
    plt.close()
else:
    print("No aspects found.")

# N-Gram extraction
def get_ngrams(token_lists, n=2):
    counter = Counter()
    for toks in token_lists:
        counter.update(list(ngrams(toks, n)))
    return counter.most_common(20)

top_bigrams = get_ngrams(tokenized_texts, 2)
top_trigrams = get_ngrams(tokenized_texts, 3)

# CSV: bigrams
with open(os.path.join(CSV_DIR, "top_bigrams.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Bigram", "Count"])
    writer.writerows([(str(bg), c) for bg, c in top_bigrams])

# CSV: trigrams
with open(os.path.join(CSV_DIR, "top_trigrams.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Trigram", "Count"])
    writer.writerows([(str(tg), c) for tg, c in top_trigrams])

# Aspect-related Ngrams
aspect_bigram_counter = Counter()
aspect_trigram_counter = Counter()

for d in all_data:
    if "Aspect" not in d or "Text" not in d:
        continue

    text = d["Text"].lower()
    toks = clean_and_tokenize(text)

    for asp in d["Aspect"]:
        if asp.lower() in text:
            aspect_bigram_counter.update(list(ngrams(toks, 2)))
            aspect_trigram_counter.update(list(ngrams(toks, 3)))

# CSV: aspect bigrams
with open(os.path.join(CSV_DIR, "aspect_bigrams.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Bigram", "Count"])
    writer.writerows([(str(bg), c) for bg, c in aspect_bigram_counter.most_common(20)])

# CSV: aspect trigrams
with open(os.path.join(CSV_DIR, "aspect_trigrams.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Trigram", "Count"])
    writer.writerows([(str(tg), c) for tg, c in aspect_trigram_counter.most_common(20)])

print("\nCSV files created in: main/csv/")

# Saving full dataset as CSV
full_dataset_csv = os.path.join(CSV_DIR, "full_dataset_cleaned.csv")

with open(full_dataset_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "Text", "Aspects", "TokenCount"])

    for idx, d in enumerate(all_data):
        text = d.get("Text", "")
        aspects = d.get("Aspect", [])
        tokens = clean_and_tokenize(text)

        writer.writerow([
            idx,
            text,
            ";".join(aspects) if isinstance(aspects, list) else aspects,
            len(tokens)
        ])

print("\n[OK] Full dataset CSV saved at:", full_dataset_csv)