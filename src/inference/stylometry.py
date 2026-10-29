"""
Pillar 2: Stylometric NLP & Naive Bayes Classifier
Evaluates writing style, linguistic tone, sensationalism, and applies
the trained Complement Naive Bayes model with TF-IDF vectorization.
Uses canonical text preprocessing to guarantee training-inference parity.
"""

from pathlib import Path
import joblib
import numpy as np
from typing import Dict, Any, Optional

from src.common.preprocessing import clean_text_advanced

SENSATIONAL_TRIGGERS = [
    "shocking", "bombshell", "unbelievable", "revealed", "exposed",
    "mainstream media", "won't believe", "secret plot", "deep state",
    "miracle cure", "they don't want you to know", "proof that", "wake up",
    "must see", "mind blowing", "conspiracy", "hidden truth", "censored"
]

try:
    from src.config import CLEAN_MODEL_PATH as MODEL_PATH, CLEAN_VECTORIZER_PATH as VECT_PATH
except ImportError:
    MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "complement_nb.pkl"
    VECT_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "tfidf_vectorizer.pkl"

_model = None
_vectorizer = None


def load_ml_assets():
    """Loads model and vectorizer with lazy caching."""
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        if not MODEL_PATH.exists() or not VECT_PATH.exists():
            raise FileNotFoundError(f"Model or vectorizer not found at {MODEL_PATH} or {VECT_PATH}. Run training pipeline first.")
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECT_PATH)
    return _model, _vectorizer


def analyze_heuristics(raw_text: str, title: str = "") -> Dict[str, Any]:
    """
    Computes clickbait and sensationalism markers across both headline and body.
    Headlines receive special attention since clickbait/hoax markers cluster heavily there.
    """
    combined_raw = f"{title} {raw_text}".strip() if title else (raw_text or "").strip()
    if not combined_raw:
        return {
            "caps_ratio": 0.0,
            "title_caps_ratio": 0.0,
            "exclamation_density": 0.0,
            "question_density": 0.0,
            "sensational_hits": [],
            "heuristic_suspicion_score": 0.0
        }

    # Title-specific capitalization (e.g. "SHOCKING BOMBSHELL: ...")
    title_caps_ratio = 0.0
    if title:
        title_letters = [c for c in title if c.isalpha()]
        if len(title_letters) > 4:
            title_caps_ratio = sum(1 for c in title_letters if c.isupper()) / len(title_letters)

    letters = [c for c in combined_raw if c.isalpha()]
    total_letters = len(letters)
    upper_letters = sum(1 for c in letters if c.isupper())
    caps_ratio = (upper_letters / total_letters) if total_letters > 0 else 0.0

    length_k = max(len(combined_raw) / 1000.0, 0.1)
    excl_count = combined_raw.count("!")
    q_count = combined_raw.count("?")
    excl_density = excl_count / length_k
    q_density = q_count / length_k

    lower_content = combined_raw.lower()
    found_triggers = [t for t in SENSATIONAL_TRIGGERS if t in lower_content]

    # Heuristic penalty score (0.0 = calm/journalistic, 1.0 = highly sensational clickbait)
    suspicion = 0.0
    # Headline in full/heavy caps is a major clickbait flag
    if title_caps_ratio > 0.35:
        suspicion += min((title_caps_ratio - 0.35) * 1.5, 0.35)
    elif caps_ratio > 0.12:
        suspicion += min((caps_ratio - 0.12) * 2.5, 0.4)

    if excl_density > 2.0:
        suspicion += min((excl_density - 2.0) * 0.08, 0.3)
    if len(found_triggers) > 0:
        suspicion += min(len(found_triggers) * 0.15, 0.45)

    return {
        "caps_ratio": round(caps_ratio, 4),
        "title_caps_ratio": round(title_caps_ratio, 4),
        "exclamation_density": round(excl_density, 2),
        "question_density": round(q_density, 2),
        "sensational_hits": found_triggers,
        "heuristic_suspicion_score": round(min(suspicion, 1.0), 3)
    }


def analyze_stylometry(text: str, title: str = "") -> Dict[str, Any]:
    """
    Executes stylometric ML analysis + heuristic clickbait auditing.
    Concatenates title and text to match the feature space used during model training.
    """
    model, vectorizer = load_ml_assets()

    heuristics = analyze_heuristics(text, title=title)
    
    # Concatenate title and body text to match training distribution (combine_features)
    full_content = f"{title} {text}".strip() if title else (text or "").strip()
    cleaned = clean_text_advanced(full_content, remove_datelines=True)

    if not cleaned.strip():
        return {
            "fake_probability": 0.5,
            "real_probability": 0.5,
            "stylometric_label": "INSUFFICIENT_TEXT",
            "heuristics": heuristics,
            "top_features": [],
            "cleaned_word_count": 0
        }

    # Vectorize
    tfidf_vec = vectorizer.transform([cleaned])
    probs = model.predict_proba(tfidf_vec)[0]
    
    # In model training: 0 = Real, 1 = Fake
    real_prob = float(probs[0])
    fake_prob = float(probs[1])

    # Extract top active vocabulary features
    feature_names = np.array(vectorizer.get_feature_names_out())
    indices = tfidf_vec.nonzero()[1]
    values = tfidf_vec.data
    
    # Sort by TF-IDF weight
    sorted_idx = np.argsort(values)[::-1][:8]
    top_words = [feature_names[indices[i]] for i in sorted_idx if i < len(indices)]

    # Blend stylometric probability with heuristic penalty
    # (Sensational heuristics pull towards Fake)
    heuristic_penalty = heuristics["heuristic_suspicion_score"] * 0.45
    adjusted_fake_prob = min(max(fake_prob + heuristic_penalty, 0.0), 1.0)
    adjusted_real_prob = 1.0 - adjusted_fake_prob

    label = "REAL_STYLE" if adjusted_real_prob >= 0.5 else "FAKE_STYLE"

    return {
        "fake_probability": round(adjusted_fake_prob, 4),
        "real_probability": round(adjusted_real_prob, 4),
        "raw_model_fake_prob": round(fake_prob, 4),
        "raw_model_real_prob": round(real_prob, 4),
        "stylometric_label": label,
        "heuristics": heuristics,
        "top_features": top_words,
        "cleaned_word_count": len(cleaned.split())
    }
