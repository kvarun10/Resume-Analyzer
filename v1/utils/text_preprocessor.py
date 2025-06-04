import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download("punkt")
nltk.download("stopwords")

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text, preserve_line=True) # 👈 Fix added here
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return tokens
