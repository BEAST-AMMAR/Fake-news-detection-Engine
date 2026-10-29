"""
Canonical Natural Language Preprocessing Module
Used across Training Pipelines, Evaluation Harnesses, and Real-Time Inference.
Guarantees consistent tokenization, entity preservation, and eliminates data leakage.
"""

import re
from typing import Optional, Set
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Pre-fetch required NLTK corpora safely
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

LEMMATIZER = WordNetLemmatizer()
STOP_WORDS: Set[str] = set(stopwords.words('english'))

# Critical 2-character acronyms and entities that must not be discarded by length filters
ALLOWED_SHORT_TOKENS: Set[str] = {
    "ai", "eu", "un", "uk", "5g", "4g", "id", "ip"
}

# Pre-compiled Regex patterns for high throughput
URL_REGEX = re.compile(r'https?://\S+|www\.\S+')
HTML_REGEX = re.compile(r'<.*?>')
# Matches datelines like "WASHINGTON (Reuters) - ", "LONDON/BEIJING (AP) — ", "(Reuters) - "
DATELINE_REGEX = re.compile(
    r'^\s*(?:[A-Za-z\s/]+)?\s*\((?:Reuters|AP|AFP|Bloomberg|CNN)\)\s*[-—–]\s*',
    re.IGNORECASE
)
# Matches agency watermarks to eliminate publisher shortcut bias (The Reuters Leak)
AGENCY_LEAK_REGEX = re.compile(r'\b(?:reuters)\b', re.IGNORECASE)
BOILERPLATE_REGEX = re.compile(r'\b(?:featured image via|image via)[^\n]*', re.IGNORECASE)
NON_ALPHANUM_REGEX = re.compile(r'[^a-zA-Z0-9\s]')
WHITESPACE_REGEX = re.compile(r'\s+')
STANDALONE_DIGITS_REGEX = re.compile(r'\b\d+\b')

# Normalize geopolitical abbreviations prior to lowercasing to disambiguate "US" (country) from "us" (pronoun)
US_ABBR_REGEX = re.compile(r'\bU\.S\.?|\bUS\b')
UK_ABBR_REGEX = re.compile(r'\bU\.K\.?|\bUK\b')


def strip_publisher_datelines(text: str) -> str:
    """
    Strips leading agency datelines (e.g. 'WASHINGTON (Reuters) - ') and explicit
    agency watermarks that induce artificial data leakage into classifiers.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    # Strip leading wire datelines
    text = DATELINE_REGEX.sub('', text)
    # Strip trailing image attribution boilerplate
    text = BOILERPLATE_REGEX.sub('', text)
    # Strip agency watermarks to prevent memorizing publisher names
    text = AGENCY_LEAK_REGEX.sub('', text)
    return text


def clean_text_advanced(text: str, remove_datelines: bool = True) -> str:
    """
    Canonical text cleaning for news articles:
    1. Dateline & publisher watermark sanitization (prevents Reuters leakage)
    2. URL & HTML tag removal
    3. Country abbreviation normalization ("U.S." / "US" -> "usa", "U.K." -> "uk")
    4. Lowercasing
    5. Normalization of standalone numeric sequences
    6. Special character removal (preserving alphanumeric and whitespace)
    7. Whitespace collapse
    8. Lemmatization with stopword filtering & short-acronym preservation (AI, EU, UN, etc.)
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    if remove_datelines:
        text = strip_publisher_datelines(text)

    # Remove URLs and HTML
    text = URL_REGEX.sub(' ', text)
    text = HTML_REGEX.sub(' ', text)

    # Disambiguate geopolitical country codes before lowercase
    text = US_ABBR_REGEX.sub('usa', text)
    text = UK_ABBR_REGEX.sub('uk', text)

    # Lowercase
    text = text.lower()

    # Normalize standalone digits so random numbers do not pollute vocabulary
    text = STANDALONE_DIGITS_REGEX.sub(' ', text)

    # Retain only alphanumeric and spaces
    text = NON_ALPHANUM_REGEX.sub(' ', text)
    text = WHITESPACE_REGEX.sub(' ', text).strip()

    tokens = text.split()
    cleaned_tokens = [
        LEMMATIZER.lemmatize(t)
        for t in tokens
        if (t not in STOP_WORDS or t in ALLOWED_SHORT_TOKENS) and (len(t) > 2 or t in ALLOWED_SHORT_TOKENS)
    ]

    return ' '.join(cleaned_tokens)


def clean_batch(texts: list) -> list:
    """Helper for parallel process pool batch execution."""
    return [clean_text_advanced(t, remove_datelines=True) for t in texts]
