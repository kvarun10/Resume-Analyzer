import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Auto-download required NLTK data if missing
def ensure_nltk_data():
    required_data = ['punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    for data in required_data:
        try:
            nltk.data.find(f'tokenizers/{data}' if 'punkt' in data else f'corpora/{data}')
        except LookupError:
            print(f"Downloading missing NLTK data: {data}")
            nltk.download(data, quiet=True)

# Ensure NLTK data is available
ensure_nltk_data()

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Clean and preprocess resume text
    
    Args:
        text (str): Raw resume text
        
    Returns:
        list: List of cleaned tokens
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Tokenize
    try:
        tokens = word_tokenize(text)
    except LookupError:
        # Fallback to simple split if punkt_tab is still missing
        print("Warning: Using simple tokenization fallback")
        tokens = text.split()
    
    # Remove stopwords and short words
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Lemmatize
    try:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    except LookupError:
        print("Warning: Lemmatization skipped - missing WordNet data")
    
    return tokens

def remove_extra_whitespace(text):
    """Remove extra whitespace and normalize text"""
    return ' '.join(text.split())

def remove_special_characters(text):
    """Remove special characters but keep basic punctuation"""
    return re.sub(r'[^\w\s.,;:!?()-]', '', text)

# Test function
if __name__ == "__main__":
    sample_text = "Hello! This is a sample resume text with special characters @#$% and numbers 123."
    cleaned = preprocess_text(sample_text)
    print("Original:", sample_text)
    print("Cleaned tokens:", cleaned)
