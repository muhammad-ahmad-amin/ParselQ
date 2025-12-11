import re
import string
import nltk
from nltk.corpus import stopwords

# Ensure stopwords downloaded (safe for repeated calls)
try:
    stopwords.words("english")
except:
    nltk.download("stopwords")


class Preprocessor:
    def __init__(self, remove_stopwords=False):
        self.remove_stopwords = remove_stopwords
        self.stop_words = set(stopwords.words("english"))

    def clean_text(self, text):
        """Apply all preprocessing steps"""

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Tokenize
        tokens = text.split()

        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]

        return " ".join(tokens)