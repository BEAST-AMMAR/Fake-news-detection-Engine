"""
Central Configuration Module
Provides centralized, strongly typed environment and operational settings
for training, inference, and service orchestration.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Storage and Artifact Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"

# Model Artifact Locations
CLEAN_MODEL_PATH = MODELS_DIR / "complement_nb.pkl"
CLEAN_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
NOISY_MODEL_PATH = MODELS_DIR / "noisy_complement_nb.pkl"
NOISY_VECTORIZER_PATH = MODELS_DIR / "noisy_tfidf_vectorizer.pkl"

# Operational & Server Settings
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

# External APIs & Credential Configurations
GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")

# Inference & Security Constraints
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "50000"))
MAX_TITLE_LENGTH = int(os.getenv("MAX_TITLE_LENGTH", "1000"))
SCRAPER_TIMEOUT_SECONDS = float(os.getenv("SCRAPER_TIMEOUT_SECONDS", "12.0"))
WIRE_CACHE_TTL_SECONDS = int(os.getenv("WIRE_CACHE_TTL_SECONDS", "600"))
MIN_CORROBORATION_OVERLAP = float(os.getenv("MIN_CORROBORATION_OVERLAP", "0.45"))
