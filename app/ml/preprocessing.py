import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# download required NLTK data (runs once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    """
    Text preprocessing for news articles.
    Focuses on normalizing text while preserving linguistic style cues
    that help distinguish real vs fake news (e.g. proper sourcing,
    formal language, factual tone).
    """
    # check if text is valid
    if not isinstance(text, str):
        return ''

    # lowercase everything
    text = text.lower()

    # remove urls, html, and weird characters
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '.', text)
    text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # remove common stop words
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]

    return ' '.join(words)


def extract_features(text):
    # extract statistics to help catch AI writing styles
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {'lexical_diversity': 0.0, 'avg_word_length': 0.0, 'punctuation_ratio': 0.0}

    words = text.split()
    if len(words) == 0:
        return {'lexical_diversity': 0.0, 'avg_word_length': 0.0, 'punctuation_ratio': 0.0}
    
    # calculate average word length
    avg_word_length = sum(len(word) for word in words) / len(words)

    # calculate lexical diversity (unique words / total words)
    unique_words = set(words)
    lexical_diversity = len(unique_words) / len(words)

    # count punctuation marks
    punctuation_count = sum(1 for char in text if char in string.punctuation)
    punctuation_ratio = punctuation_count / len(text)

    return {
        'lexical_diversity': lexical_diversity,
        'avg_word_length': avg_word_length,
        'punctuation_ratio': punctuation_ratio
    }
