import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer
import numpy as np

nltk.download('stopwords')

# Config
TRAIN_PATH = "task-dataset/track_b/subtask_1/eng/eng_environmental_protection_train_task1.jsonl"
DEV_PATH   = "task-dataset/track_b/subtask_1/eng/eng_environmental_protection_dev_task1.jsonl"
PLOT_DIR   = "plots"

os.makedirs(PLOT_DIR, exist_ok=True)

stop_words = set(stopwords.words("english"))

tokenizer = RegexpTokenizer(r'\w+')

# Loading Data
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

# Text Cleaning and Tokenization
def clean_and_tokenize(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]  # remove stopwords
    return tokens

tokenized_texts = [clean_and_tokenize(t) for t in texts]
token_lengths = [len(toks) for toks in tokenized_texts]

# Token Length Distribution
plt.figure(figsize=(8, 5))
plt.hist(token_lengths, bins=40)
plt.xlabel("Token Count")
plt.ylabel("Frequency")
plt.title("Token Length Distribution")
plt.savefig(os.path.join(PLOT_DIR, "token_length_distribution.pdf"))
plt.close()

print("Token length statistics:")
print(f" Mean:   {np.mean(token_lengths):.2f}")
print(f" Median: {np.median(token_lengths):.2f}")
print(f" Min:    {np.min(token_lengths)}")
print(f" Max:    {np.max(token_lengths)}")

# Most Frequent Tokens
token_counter = Counter()
for toks in tokenized_texts:
    token_counter.update(toks)

top_tokens = token_counter.most_common(30)
tokens, freqs = zip(*top_tokens)

plt.figure(figsize=(10, 6))
plt.bar(tokens, freqs)
plt.xticks(rotation=75, ha="right")
plt.title("Top 30 Most Frequent Tokens")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "top_tokens.pdf"))
plt.close()

# Most Frequent Aspect Terms
if all_aspects:
    aspect_counter = Counter(all_aspects)
    top_aspects = aspect_counter.most_common(20)
    aspects, counts = zip(*top_aspects)

    plt.figure(figsize=(10, 6))
    plt.bar(aspects, counts)
    plt.xticks(rotation=75, ha="right")
    plt.title("Most Frequent Aspect Terms")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "aspect_frequency.pdf"))
    plt.close()
else:
    print("No aspects found in data.")

# Bigrams and Trigrams
def get_ngrams(token_lists, n=2):
    counter = Counter()
    for toks in token_lists:
        counter.update(list(ngrams(toks, n)))
    return counter.most_common(20)

top_bigrams = get_ngrams(tokenized_texts, 2)
top_trigrams = get_ngrams(tokenized_texts, 3)

print("\nTop 10 bigrams:")
for bg in top_bigrams[:10]:
    print(bg)

print("\nTop 10 trigrams:")
for tg in top_trigrams[:10]:
    print(tg)

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

print("\nTop aspect-related bigrams:", aspect_bigram_counter.most_common(10))
print("Top aspect-related trigrams:", aspect_trigram_counter.most_common(10))