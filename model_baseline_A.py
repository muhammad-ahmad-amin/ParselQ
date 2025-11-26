import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from preprocess import Preprocessor

class BaselineModelA:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000)

        os.makedirs("plots", exist_ok=True)
        os.makedirs("results", exist_ok=True)

    def train(self, train_df):
        print("[INFO] Vectorizing training text...")
        X_train = self.vectorizer.fit_transform(train_df["clean_text"])
        y_train = train_df["label"]

        print("[INFO] Training Logistic Regression...")
        self.model.fit(X_train, y_train)
        print("[INFO] Baseline A training complete")

    def evaluate(self, test_df, train_df):
        print("[INFO] Evaluating model...")
        X_test = self.vectorizer.transform(test_df["clean_text"])
        y_test = test_df["label"]

        preds = self.model.predict(X_test)

        # Accuracy & classification report
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds)
        print("\n===== BASELINE A REPORT =====")
        print(report)
        print("Accuracy:", acc)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        text_str = "===== BASELINE A REPORT =====\n\n" + report + f"\nAccuracy: {acc:.4f}"
        ax.text(0, 1, text_str, fontsize=10, va='top', family='monospace')
        results_pdf_path = os.path.join("results", "baselineA_results.pdf")
        fig.savefig(results_pdf_path, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Classification report saved to {results_pdf_path}")

        # Confusion matrix
        cm = confusion_matrix(y_test, preds, labels=self.model.classes_)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.model.classes_,
                    yticklabels=self.model.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Baseline A: Confusion Matrix")
        cm_pdf_path = os.path.join("plots", "baselineA_confusion_matrix.pdf")
        plt.savefig(cm_pdf_path, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Confusion matrix saved to {cm_pdf_path}")
        plt.figure(figsize=(10, 5))
        sns.countplot(y=train_df['label'], order=train_df['label'].value_counts().index, palette="viridis")
        plt.title("Training Label Distribution")
        plt.xlabel("Count")
        plt.ylabel("Label")
        plt.tight_layout()
        label_dist_path = os.path.join("plots", "label_distribution.pdf")
        plt.savefig(label_dist_path)
        plt.close()
        print(f"[INFO] Label distribution plot saved to {label_dist_path}")
        train_df['text_length'] = train_df['clean_text'].apply(lambda x: len(x.split()))
        plt.figure(figsize=(8, 5))
        sns.histplot(train_df['text_length'], bins=30, kde=True, color="purple")
        plt.title("Training Text Length Distribution")
        plt.xlabel("Number of words")
        plt.ylabel("Frequency")
        plt.tight_layout()
        text_length_path = os.path.join("plots", "text_length_distribution.pdf")
        plt.savefig(text_length_path)
        plt.close()
        print(f"[INFO] Text length distribution plot saved to {text_length_path}")


if __name__ == "__main__":
    # Load CSV
    df = pd.read_csv("main/csv/loaded_data.csv")

    # Check CSV columns
    print("[INFO] CSV Columns:", df.columns.tolist())
    if 'Text' not in df.columns or 'Aspects' not in df.columns:
        raise KeyError("CSV must contain 'Text' and 'Aspects' columns.")

    # Drop rows with missing Text or Aspects
    df = df.dropna(subset=['Text', 'Aspects']).reset_index(drop=True)

    # Initialize preprocessor
    preprocessor = Preprocessor(remove_stopwords=True)
    df['clean_text'] = df['Text'].apply(preprocessor.clean_text)
    df['label'] = df['Aspects']

    # Handle rare labels for stratified split
    label_counts = df['label'].value_counts()
    stratify_labels = label_counts[label_counts >= 2].index.tolist()
    df_strat = df[df['label'].isin(stratify_labels)]
    df_rare = df[~df['label'].isin(stratify_labels)]

    # Stratified split for common labels
    train_strat, test_strat = train_test_split(
        df_strat, test_size=0.2, random_state=42, stratify=df_strat['label']
    )

    # Combine with rare labels (all in training)
    train_df = pd.concat([train_strat, df_rare]).reset_index(drop=True)
    test_df = test_strat.reset_index(drop=True)

    print(f"[INFO] Training samples: {len(train_df)}, Testing samples: {len(test_df)}")
    print("Label distribution in train:\n", train_df['label'].value_counts())
    print("Label distribution in test:\n", test_df['label'].value_counts())

    # Create and train model
    model = BaselineModelA()
    model.train(train_df)

    # Evaluate
    model.evaluate(test_df, train_df)