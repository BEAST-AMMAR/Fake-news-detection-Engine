# Project Beta: Fake News Detection System
## Naive Bayes Classification for Misinformation Identification

---

**Document Version:** 2.0
**Date:** August 2026
**Classification:** Internal — Hackathon Development Kit
**Prepared By:** Core AI/ML Engineering Division
**Project Code:** NB-FND-2026
**Team Assignment:** Team Beta (4 Members)
**Dataset Source:** Kaggle — Fake News Detection Datasets (emineyetm)
**Dataset URL:** https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Product Requirements Document (PRD)](#2-product-requirements-document-prd)
3. [Technical Requirements Document (TRD)](#3-technical-requirements-document-trd)
4. [Project Development Document (PD)](#4-project-development-document-pd)
5. [Implementation Plan — Phase 1](#5-implementation-plan--phase-1-mvp-pipeline)
6. [Code Implementation Kit](#6-code-implementation-kit)
7. [Future Roadmap](#7-future-roadmap--phase-2)
8. [Team Structure](#8-team-structure--raci-matrix)
9. [Risk Assessment & Mitigation](#9-risk-assessment--mitigation)
10. [Appendices](#10-appendices)

---

## 1. Executive Summary

### 1.1 Problem Statement
The global information ecosystem faces an existential threat from fabricated news articles designed to manipulate public opinion, influence elections, and destabilize democratic institutions. Traditional fact-checking cannot scale to the velocity of modern information dissemination. An automated, lightweight, and interpretable detection system is urgently needed to flag potentially fake news at scale.

### 1.2 Project Objective
Develop a **Naive Bayes-based text classification system** capable of distinguishing authentic news from fabricated articles with >90% accuracy. The system leverages the probabilistic nature of Naive Bayes to provide transparent, explainable predictions — critical for journalistic and regulatory applications where black-box models are unacceptable.

### 1.3 Success Criteria

| KPI | Target | Measurement Method |
|-----|--------|---------------------|
| Classification Accuracy | ≥ 90% | Hold-out test set evaluation |
| F1-Score (Macro) | ≥ 0.88 | sklearn `f1_score` macro average |
| ROC-AUC Score | ≥ 0.92 | `roc_auc_score` on test data |
| Precision (Fake Class) | ≥ 0.85 | Minimize false accusations |
| Recall (Fake Class) | ≥ 0.85 | Catch most fake articles |
| Training Time | < 5 minutes | Full pipeline on standard CPU |
| Model Size | < 15 MB | Serialized `.pkl` file size |

### 1.4 Scope & Constraints

**In Scope (Phase 1 — Current):**
- Dataset acquisition from Kaggle (`emineyetm/fake-news-detection-datasets`)
- Advanced text preprocessing (special character/URL removal, lemmatization)
- Combined title + body feature engineering
- Naive Bayes model training (MultinomialNB, ComplementNB)
- Hyperparameter optimization (GridSearchCV)
- Comprehensive evaluation with per-class metrics
- Confusion Matrix, ROC Curve, Learning Curve, Word Cloud visualizations
- Model serialization and export

**Out of Scope (Future Phases):**
- Source credibility scoring (domain reputation)
- Image-based fake news detection (deepfakes)
- Real-time social media stream processing
- Multi-lingual fake news detection
- Network propagation analysis (how fake news spreads)

---

## 2. Product Requirements Document (PRD)

### 2.1 Product Vision
Build an **interpretable fake news detection engine** that empowers journalists, fact-checkers, and platform moderators to identify potentially fabricated articles before they reach mass audiences. Unlike deep learning black boxes, our Naive Bayes approach provides clear probabilistic reasoning for every flag — enabling human reviewers to make informed decisions.

### 2.2 User Personas

#### Persona 1: Fact-Check Journalist (Priya, The Hindu)
- **Needs:** Rapidly screen incoming news tips for credibility
- **Pain Points:** Overwhelmed by volume; cannot manually verify every article
- **Success Metric:** 90% of fake articles flagged for secondary review

#### Persona 2: Social Media Platform Moderator (Meta/YouTube)
- **Needs:** Batch-process submitted news content at scale
- **Pain Points:** GPU-dependent models are too expensive for billions of posts
- **Success Metric:** Process 50K articles/hour on a single server

#### Persona 3: Election Commission Official
- **Needs:** Monitor political news for coordinated misinformation campaigns
- **Pain Points:** Need explainable decisions for legal defensibility
- **Success Metric:** Every flagged article comes with word-level probability evidence

### 2.3 Functional Requirements (FR)

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-1 | Load `True.csv` and `Fake.csv`, assign labels, merge | P0 | Unified DataFrame; no data loss |
| FR-2 | Combine title and body text for feature richness | P0 | Combined text column created; title weight preserved |
| FR-3 | Remove URLs, special characters, and web artifacts | P0 | No URL/markup residue in cleaned text |
| FR-4 | Lemmatization and advanced stopword removal | P1 | Words reduced to root forms; domain-specific stopwords filtered |
| FR-5 | TF-IDF vectorization with n-grams (1,2) | P0 | Sparse matrix generated; bigram features captured |
| FR-6 | Train MultinomialNB baseline | P0 | Model trained; baseline metrics recorded |
| FR-7 | Train ComplementNB primary model | P0 | Model trained; handles class imbalance |
| FR-8 | Hyperparameter tuning via GridSearchCV | P1 | Optimal alpha identified; CV scores documented |
| FR-9 | Generate all evaluation metrics | P0 | Accuracy, Precision, Recall, F1, ROC-AUC computed |
| FR-10 | Confusion Matrix visualization | P0 | Heatmap with Fake/Real labels |
| FR-11 | ROC Curve for model comparison | P1 | All three models plotted; AUC annotated |
| FR-12 | Learning Curve analysis | P1 | Overfitting/underfitting visually detected |
| FR-13 | Word Cloud generation (Fake vs Real) | P2 | Top distinctive words visualized per class |
| FR-14 | Model export (`.pkl` format) | P2 | Reloadable artifacts created |
| FR-15 | Classification report per class | P0 | Precision, Recall, F1 for Fake and Real |

### 2.4 Non-Functional Requirements (NFR)

| ID | Requirement | Priority | Target |
|----|-------------|----------|--------|
| NFR-1 | **Performance** — Full pipeline < 5 min on CPU | P0 | Intel i5 / 8GB RAM |
| NFR-2 | **Scalability** — Handle up to 50K articles | P1 | Sparse matrix optimization |
| NFR-3 | **Interpretability** — Top features per class extractable | P1 | `feature_log_prob_` accessible |
| NFR-4 | **Reliability** — 5-Fold CV with std < 0.02 | P0 | Consistent cross-validation |
| NFR-5 | **Maintainability** — Modular, PEP-8 compliant code | P1 | Clear function separation |
| NFR-6 | **Reproducibility** — Fixed `random_state=42` | P1 | Deterministic results |

### 2.5 Data Requirements

**Primary Source:** Kaggle — Fake News Detection Datasets (IEEE-derived mirror)
**Author:** emineyetm
**URL:** https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets
**Underlying Source:** Reuters (real articles) + Politifact-flagged articles (fake), consistent with the IEEE DataPort fake-news corpus
**Format:** Two CSV files — `True.csv` + `Fake.csv` (ISOT-style schema; commonly distributed together with this dataset)
**Size:** ~12,600–45,000 articles depending on which file variant is downloaded (verify exact row count after download, since Kaggle mirrors of this corpus vary slightly in packaging)
**Columns:** `title`, `text`, `subject`, `date` (label assigned during ingestion: 0 = Real, 1 = Fake)
**Access:** Requires a free Kaggle account + Kaggle API token (`kaggle.json`), or manual download via the "Download" button on the dataset page
**Data Quality:**
- Professional news articles (Reuters for real; various flagged sources for fake)
- Minimal missing values
- Class balance: approximately 50–50
- Text length: 500–5,000 characters per article

**Download methods:**
```bash
# Option A: Kaggle CLI (requires ~/.kaggle/kaggle.json API credentials)
pip install kaggle
kaggle datasets download -d emineyetm/fake-news-detection-datasets
unzip fake-news-detection-datasets.zip -d data/raw/

# Option B: Manual
# Visit the Kaggle URL above → click "Download" → extract into data/raw/
```

**Note:** Because this is a Kaggle-hosted dataset, it requires authentication (unlike a raw GitHub CSV link). Confirm the exact filenames after extraction — this dataset is commonly packaged as `True.csv` + `Fake.csv`, but verify against `data/raw/` contents before running the pipeline, and adjust the `load_and_merge_datasets()` paths in Section 6 accordingly.

---

## 3. Technical Requirements Document (TRD)

### 3.1 Technology Stack

| Layer | Technology | Version | Justification |
|-------|-----------|---------|---------------|
| Language | Python | 3.10+ | ML industry standard |
| Data Processing | pandas | 2.0+ | Large DataFrame handling |
| Numerical Computing | numpy | 1.24+ | Array operations |
| ML Framework | scikit-learn | 1.3+ | Naive Bayes, GridSearch, metrics |
| Visualization | matplotlib | 3.7+ | Publication-quality plots |
| Visualization | seaborn | 0.12+ | Statistical heatmaps |
| NLP | NLTK | 3.8+ | Lemmatization, stopwords |
| Word Cloud | wordcloud | 1.9+ | Fake vs Real word visualization |
| Dataset Access | kaggle (CLI) | latest | Programmatic dataset download |
| Environment | Jupyter / VS Code | latest | Interactive development |

### 3.2 System Architecture

```
                    INPUT LAYER
        Raw News Articles (True.csv + Fake.csv via Kaggle)
                         |
                         v
        DATA INGESTION & MERGE
   Load True.csv --> Load Fake.csv --> Concatenate + Label Assignment
                         |
                         v
        PREPROCESSING PIPELINE
   URL/Special-char removal --> Lowercase + Punctuation removal
                --> Lemmatization + Stopword removal (NLTK WordNet)
                         |
                         v
        FEATURE ENGINEERING
   Title + Body Merge --> TF-IDF Vectorizer --> N-gram Features (1,2)
                         |
                         v
        MODEL LAYER
      MultinomialNB (Baseline)      ComplementNB (Primary)
                         |
                         v
        OPTIMIZATION & EVALUATION
   GridSearchCV (alpha) --> 5-Fold CV --> Metrics + Visualizations
```

### 3.3 Algorithm Specifications

#### 3.3.1 Multinomial Naive Bayes
- **Type:** Probabilistic generative classifier for discrete features
- **Mechanism:** Models word frequency distributions as multinomial
- **Best For:** Text classification with count/TF-IDF features
- **Hyperparameter:** `alpha` (Laplace smoothing, default 1.0)
- **Limitation:** Biased toward majority class in imbalanced settings

#### 3.3.2 Complement Naive Bayes
- **Type:** Modified MultinomialNB for imbalanced datasets
- **Mechanism:** Estimates parameters using the complement of each class
- **Best For:** When fake vs. real article distribution is skewed
- **Advantage:** Outperforms MultinomialNB on imbalanced text data

#### 3.3.3 TF-IDF with N-grams
- **Unigrams:** Single words (`president`, `election`)
- **Bigrams:** Two-word phrases (`fake news`, `breaking story`)
- **Parameters:**
  - `ngram_range=(1, 2)`
  - `max_features=10000`
  - `min_df=2` (ignore rare terms)
  - `max_df=0.90` (ignore terms in >90% of docs)
  - `sublinear_tf=True` (logarithmic term frequency)

### 3.4 Data Flow Diagram

```
True.csv + Fake.csv (Kaggle download)
        |
        v
Load --> Add label column --> Concatenate --> Single DataFrame
        |
        v
EDA: shape, nulls, class balance, text length distribution
        |
        v
Combine title + text --> 'title_text' column
        |
        v
Clean: URL removal --> lowercase --> punctuation --> lemmatize --> stopwords
        |
        v
Train-Test Split (80-20, stratify=y, random_state=42)
        |
        v
TF-IDF Fit (train only) --> Transform (train + test)
        |
        v
Model Training: MultinomialNB + ComplementNB + GridSearchCV
        |
        v
Evaluation: Metrics + Confusion Matrix + ROC + Learning Curve + Word Cloud
        |
        v
Export: .pkl files + reports + visualizations
```

### 3.5 Security & Privacy Considerations
- **Data Source:** Public Kaggle dataset; no PII in news articles
- **Credential Handling:** Kaggle API token (`kaggle.json`) must never be committed to version control; store outside the repo or use environment variables
- **Input Sanitization:** Regex-based URL/special-character stripping prevents downstream injection issues
- **Model Integrity:** Checksum verification for serialized models
- **Bias Awareness:** Dataset may contain political bias; document limitations in the final report
- **Future:** Adversarial input detection; rate limiting on API

---

## 4. Project Development Document (PD)

### 4.1 Development Methodology
**Approach:** Agile-Scrum (compressed for a 40-minute hackathon)
**Sprint 0:** Planning & Architecture (this document)
**Sprint 1:** Data Acquisition & Preprocessing (0–15 min)
**Sprint 2:** Feature Engineering & Model Training (15–30 min)
**Sprint 3:** Evaluation, Visualization & Export (30–40 min)

### 4.2 Milestone Timeline

| Milestone | Deliverable | Owner | Duration | Status |
|-----------|-------------|-------|----------|--------|
| M0 | Kaggle dataset downloaded & extracted | Data Engineer | 2 min | Planned |
| M1 | Both CSVs loaded and merged | Data Engineer | 3 min | Planned |
| M2 | EDA completed (class balance, text stats) | Data Engineer | 3 min | Planned |
| M3 | Title+Body combined and cleaned | ML Engineer | 5 min | Planned |
| M4 | TF-IDF vectorizer fitted | ML Engineer | 3 min | Planned |
| M5 | MultinomialNB baseline trained | ML Engineer | 3 min | Planned |
| M6 | ComplementNB + GridSearch trained | ML Engineer | 7 min | Planned |
| M7 | All metrics computed and logged | Data Scientist | 5 min | Planned |
| M8 | All visualizations generated | Software Engineer | 6 min | Planned |
| M9 | Model exported and documented | Software Engineer | 3 min | Planned |

### 4.3 Environment Setup

```bash
# Required Packages
pip install pandas numpy scikit-learn matplotlib seaborn nltk wordcloud kaggle

# NLTK Data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Kaggle API setup (place kaggle.json in ~/.kaggle/)
kaggle datasets download -d emineyetm/fake-news-detection-datasets
unzip fake-news-detection-datasets.zip -d data/raw/
```

### 4.4 Directory Structure

```
project-beta-fake-news/
|-- data/
|   |-- raw/
|   |   |-- True.csv
|   |   |-- Fake.csv
|   |-- processed/
|   |   |-- merged_news.csv
|   |   |-- cleaned_news.csv
|-- notebooks/
|   |-- 01_pipeline_development.ipynb
|-- src/
|   |-- data_loader.py
|   |-- preprocessor.py
|   |-- feature_engineer.py
|   |-- model_trainer.py
|   |-- evaluator.py
|   |-- visualizer.py
|-- models/
|   |-- multinomial_nb.pkl
|   |-- complement_nb.pkl
|   |-- tfidf_vectorizer.pkl
|-- outputs/
|   |-- confusion_matrix.png
|   |-- roc_curve.png
|   |-- learning_curve.png
|   |-- wordcloud_fake.png
|   |-- wordcloud_real.png
|   |-- classification_report.txt
|-- docs/
|   |-- PRD_TRD_PD.md
|-- requirements.txt
```

---

## 5. Implementation Plan — Phase 1: MVP Pipeline

### 5.1 Phase 1 Objective
Deliver a complete end-to-end fake news detection pipeline: from Kaggle dataset acquisition through advanced preprocessing, feature engineering, model training, hyperparameter optimization, comprehensive evaluation, and visualization. This MVP establishes the foundation for production deployment.

### 5.2 Step-by-Step Implementation

#### Step 0: Kaggle Dataset Acquisition (2 minutes)
**Objective:** Authenticate with Kaggle and download the dataset.
**Actions:**
- Generate a Kaggle API token from Account Settings → create `kaggle.json`
- Place it at `~/.kaggle/kaggle.json` (or set `KAGGLE_CONFIG_DIR`)
- Run `kaggle datasets download -d emineyetm/fake-news-detection-datasets`
- Extract into `data/raw/`

**Success Criteria:**
- `True.csv` and `Fake.csv` present in `data/raw/`
- File sizes non-zero; readable by pandas

#### Step 1: Dataset Ingestion & Merging (3 minutes)
**Objective:** Load both CSVs, assign labels, and merge into a single DataFrame.
**Actions:**
- Load `Fake.csv`, add `label=1` (Fake)
- Load `True.csv`, add `label=0` (Real)
- Concatenate vertically using `pd.concat`
- Verify combined shape and class distribution
- Shuffle the combined DataFrame

**Success Criteria:**
- Single DataFrame with columns: `['title', 'text', 'subject', 'date', 'label']`
- Class distribution approximately balanced
- No data loss during merge

#### Step 2: Exploratory Data Analysis (3 minutes)
**Objective:** Understand dataset characteristics before modeling.
**Actions:**
- Check shape, dtypes, missing values
- Analyze class distribution
- Compute text length statistics (mean, median, std)
- Examine subject categories
- Check for duplicate articles

**Key Insights to Document:**
- Average article length (characters and words)
- Most common subjects in fake vs. real articles
- Date range of articles

#### Step 3: Feature Engineering — Title + Body Merge (2 minutes)
**Objective:** Create a rich text feature by combining title and body.
**Rationale:** Fake news titles are often more sensational than the body. Combining both captures the full narrative.
**Actions:**
```python
df['title_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
```
**Success Criteria:**
- New `title_text` column created
- No NaN values in combined column

#### Step 4: Advanced Text Preprocessing (8 minutes)
**Objective:** Clean and normalize news article text for machine learning.
**Actions:**
1. **URL Removal:** Regex-based URL stripping
2. **Lowercasing:** Convert to lowercase for consistency
3. **Punctuation Removal:** Strip special characters (keep alphanumeric + spaces)
4. **Lemmatization:** Reduce words to root forms using `WordNetLemmatizer`
5. **Stopword Removal:** Filter common English stopwords
6. **Extra Whitespace Normalization:** Collapse multiple spaces

**Why Lemmatization over Stemming?**
- Lemmatization produces valid dictionary words (e.g., "running" → "run")
- Better for news articles where word meaning matters
- Stemming can produce non-words ("running" → "runn") which hurt TF-IDF

**Success Criteria:**
- 100% of articles processed without errors
- No URL residue remaining
- Text length reduction documented

#### Step 5: Train-Test Split (2 minutes)
**Objective:** Create stratified train/test sets preserving class balance.
**Parameters:**
- `test_size=0.2` (20% test)
- `random_state=42`
- `stratify=y` (maintain fake/real ratio)

**Success Criteria:**
- Class proportions identical in both splits

#### Step 6: TF-IDF Vectorization (3 minutes)
**Objective:** Convert cleaned text to numerical feature vectors.
**Parameters:**
```python
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),          # Unigrams + Bigrams
    max_features=10000,          # Larger vocab for news articles
    min_df=2,                    # Ignore rare terms
    max_df=0.90,                 # Ignore terms in >90% docs
    stop_words='english',        # Built-in stopwords
    sublinear_tf=True            # Logarithmic scaling
)
```
**Success Criteria:**
- Sparse matrix generated
- Vocabulary size documented
- Fit ONLY on training data (no leakage)

#### Step 7: Model Training (10 minutes)
**7A. MultinomialNB (Baseline)**
```python
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_tfidf, y_train)
```

**7B. ComplementNB (Primary)**
```python
from sklearn.naive_bayes import ComplementNB
cnb = ComplementNB(alpha=1.0)
cnb.fit(X_train_tfidf, y_train)
```

**7C. Hyperparameter Optimization**
```python
param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]}
grid = GridSearchCV(ComplementNB(), param_grid, cv=5, scoring='f1_macro')
grid.fit(X_train_tfidf, y_train)
best_model = grid.best_estimator_
```

**Success Criteria:**
- All models trained successfully
- GridSearch identifies best alpha
- 5-Fold CV scores documented

#### Step 8: Comprehensive Evaluation (5 minutes)

| Metric | Purpose | Target |
|--------|---------|--------|
| Accuracy | Overall correctness | > 90% |
| Precision (Fake) | Minimize false accusations | > 85% |
| Recall (Fake) | Catch most fake articles | > 85% |
| F1-Score (Macro) | Balanced performance | > 0.88 |
| ROC-AUC | Discriminative power | > 0.92 |
| Confusion Matrix | Error pattern analysis | Visual |

**Actions:**
- Predict on test set
- Compute all metrics
- Generate classification report
- Log results

#### Step 9: Visualization Generation (6 minutes)
**Visualizations:**
1. **Confusion Matrix Heatmap** — Error patterns per class
2. **ROC Curve** — Model comparison with AUC scores
3. **Learning Curve** — Overfitting/underfitting detection
4. **Word Cloud (Fake)** — Most common words in fake articles
5. **Word Cloud (Real)** — Most common words in real articles

**Word Cloud Rationale:**
- Fake news often uses sensational language ("shocking", "breaking", "exclusive")
- Real news uses formal, neutral language
- Word clouds provide immediate visual insight into linguistic differences

#### Step 10: Model Export & Documentation (3 minutes)
**Actions:**
- Serialize best model: `joblib.dump(best_model, 'models/complement_nb.pkl')`
- Serialize vectorizer: `joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')`
- Save all visualizations as 300 DPI PNG
- Export classification report to text file
- Document final hyperparameters

---

## 6. Code Implementation Kit

### 6.1 Complete Pipeline Script

```python
"""
Project Beta: Fake News Detection
Team: Beta (4 Members)
Model: Naive Bayes (ComplementNB + MultinomialNB)
Dataset: Kaggle - emineyetm/fake-news-detection-datasets
Pipeline: Load -> Merge -> Preprocess -> TF-IDF -> Train -> Evaluate -> Visualize -> Export
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import warnings
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, classification_report, roc_curve, auc,
                              roc_auc_score)

warnings.filterwarnings('ignore')
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 10000
NGRAM_RANGE = (1, 2)
CV_FOLDS = 5

# ============================================================
# STEP 1: DATASET LOADING & MERGING
# ============================================================
def load_and_merge_datasets(fake_path, true_path):
    """
    Load Fake.csv and True.csv, assign labels, and merge.
    label: 0 = Real, 1 = Fake
    """
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df['label'] = 1
    true_df['label'] = 0

    df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print('=' * 60)
    print('DATASET OVERVIEW')
    print('=' * 60)
    print(f'Total Shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    print('\nClass Distribution:')
    print(df['label'].value_counts())
    print('\nMissing Values:')
    print(df.isnull().sum())
    print(f'\nDuplicate Rows: {df.duplicated().sum()}')
    return df

# ============================================================
# STEP 2: FEATURE ENGINEERING - TITLE + BODY
# ============================================================
def combine_features(df):
    """Combine title and text for richer features."""
    df = df.copy()
    df['title_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    print(f"\nCombined title_text length: {df['title_text'].str.len().mean():.1f} chars (avg)")
    return df

# ============================================================
# STEP 3: ADVANCED TEXT PREPROCESSING
# ============================================================
def clean_text_advanced(text):
    """
    Advanced cleaning for news articles:
    1. URL removal
    2. Lowercase
    3. Special character removal
    4. Lemmatization
    5. Stopword removal
    6. Whitespace normalization
    """
    if pd.isna(text):
        return ''

    text = text.lower()
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

def preprocess_dataframe(df, text_col='title_text', label_col='label'):
    """Apply advanced cleaning to entire DataFrame."""
    df = df.copy()
    print('\nPreprocessing text... (this may take a moment)')
    df['cleaned_text'] = df[text_col].apply(clean_text_advanced)
    df = df[df['cleaned_text'].str.len() > 0]
    print(f'After cleaning: {len(df)} samples remain')
    return df

# ============================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================
def split_data(df, text_col='cleaned_text', label_col='label'):
    """Stratified split to maintain class balance."""
    X = df[text_col]
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f'\nTrain set: {len(X_train)} samples')
    print(f'Test set: {len(X_test)} samples')
    return X_train, X_test, y_train, y_test

# ============================================================
# STEP 5: TF-IDF VECTORIZATION
# ============================================================
def vectorize_text(X_train, X_test):
    """Convert text to TF-IDF features."""
    vectorizer = TfidfVectorizer(
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
        min_df=2,
        max_df=0.90,
        stop_words='english',
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f'\nTF-IDF Matrix Shape: {X_train_tfidf.shape}')
    print(f'Vocabulary Size: {len(vectorizer.vocabulary_)}')
    return vectorizer, X_train_tfidf, X_test_tfidf

# ============================================================
# STEP 6: MODEL TRAINING
# ============================================================
def train_models(X_train_tfidf, y_train):
    """Train baseline and primary models with CV."""

    print('\n' + '=' * 60)
    print('TRAINING: MultinomialNB (Baseline)')
    print('=' * 60)
    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_train_tfidf, y_train)
    mnb_cv = cross_val_score(mnb, X_train_tfidf, y_train, cv=CV_FOLDS, scoring='f1_macro')
    print(f'5-Fold CV F1-Macro: {mnb_cv.mean():.4f} (+/- {mnb_cv.std():.4f})')

    print('\n' + '=' * 60)
    print('TRAINING: ComplementNB (Primary)')
    print('=' * 60)
    cnb = ComplementNB(alpha=1.0)
    cnb.fit(X_train_tfidf, y_train)
    cnb_cv = cross_val_score(cnb, X_train_tfidf, y_train, cv=CV_FOLDS, scoring='f1_macro')
    print(f'5-Fold CV F1-Macro: {cnb_cv.mean():.4f} (+/- {cnb_cv.std():.4f})')

    print('\n' + '=' * 60)
    print('HYPERPARAMETER TUNING: GridSearchCV (alpha)')
    print('=' * 60)
    param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]}
    grid = GridSearchCV(ComplementNB(), param_grid, cv=CV_FOLDS,
                         scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train_tfidf, y_train)
    best_model = grid.best_estimator_
    print(f'Best Alpha: {grid.best_params_["alpha"]}')
    print(f'Best CV F1-Macro: {grid.best_score_:.4f}')

    return mnb, cnb, best_model, grid

# ============================================================
# STEP 7: MODEL EVALUATION
# ============================================================
def evaluate_model(model, X_test_tfidf, y_test, model_name, class_names=['Real', 'Fake']):
    """Comprehensive evaluation with all metrics."""
    y_pred = model.predict(X_test_tfidf)
    y_prob = model.predict_proba(X_test_tfidf)[:, 1]

    print('\n' + '=' * 60)
    print(f'EVALUATION: {model_name}')
    print('=' * 60)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f'Accuracy:  {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall:    {rec:.4f}')
    print(f'F1-Score:  {f1:.4f}')
    print(f'ROC-AUC:   {roc_auc:.4f}')
    print(f'\nClassification Report:')
    print(classification_report(y_test, y_pred, target_names=class_names))

    return y_pred, y_prob, {'accuracy': acc, 'precision': prec, 'recall': rec,
                             'f1': f1, 'roc_auc': roc_auc}

# ============================================================
# STEP 8: VISUALIZATION
# ============================================================
def plot_confusion_matrix(y_test, y_pred, class_names=['Real', 'Fake'], save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Fake News Detection', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'\nConfusion Matrix saved: {save_path}')

def plot_roc_curve(y_test, y_prob_mnb, y_prob_cnb, y_prob_best, save_path='roc_curve.png'):
    plt.figure(figsize=(10, 8))
    for y_prob, name, color in [(y_prob_mnb, 'MultinomialNB', 'orange'),
                                 (y_prob_cnb, 'ComplementNB', 'green'),
                                 (y_prob_best, 'Best ComplementNB', 'blue')]:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison - Fake News Detection', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'ROC Curve saved: {save_path}')

def plot_learning_curve(model, X, y, model_name, save_path='learning_curve.png'):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=CV_FOLDS, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='f1_macro'
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes, test_mean, 'o-', color='green', label='Cross-Validation Score')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='green')
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('F1-Macro Score', fontsize=12)
    plt.title(f'Learning Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Learning Curve saved: {save_path}')

def plot_wordclouds(df, text_col='cleaned_text', label_col='label',
                     save_fake='wordcloud_fake.png', save_real='wordcloud_real.png'):
    fake_text = ' '.join(df[df[label_col] == 1][text_col].dropna())
    real_text = ' '.join(df[df[label_col] == 0][text_col].dropna())

    wc_fake = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(fake_text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc_fake, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud - FAKE News Articles', fontsize=16, fontweight='bold', color='darkred')
    plt.tight_layout()
    plt.savefig(save_fake, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Fake Word Cloud saved: {save_fake}')

    wc_real = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(real_text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc_real, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud - REAL News Articles', fontsize=16, fontweight='bold', color='darkblue')
    plt.tight_layout()
    plt.savefig(save_real, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Real Word Cloud saved: {save_real}')

# ============================================================
# STEP 9: MODEL EXPORT
# ============================================================
def export_model(model, vectorizer, model_path='complement_nb.pkl', vec_path='tfidf_vectorizer.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f'\nModel exported: {model_path}')
    print(f'Vectorizer exported: {vec_path}')

# ============================================================
# MAIN EXECUTION PIPELINE
# ============================================================
def main_pipeline(fake_path, true_path):
    """End-to-end execution."""
    df = load_and_merge_datasets(fake_path, true_path)
    df = combine_features(df)
    df = preprocess_dataframe(df)

    X_train, X_test, y_train, y_test = split_data(df)
    vectorizer, X_train_tfidf, X_test_tfidf = vectorize_text(X_train, X_test)
    mnb, cnb, best_model, grid = train_models(X_train_tfidf, y_train)

    y_pred_mnb, y_prob_mnb, metrics_mnb = evaluate_model(mnb, X_test_tfidf, y_test, "MultinomialNB")
    y_pred_cnb, y_prob_cnb, metrics_cnb = evaluate_model(cnb, X_test_tfidf, y_test, "ComplementNB")
    y_pred_best, y_prob_best, metrics_best = evaluate_model(best_model, X_test_tfidf, y_test, "Best ComplementNB")

    plot_confusion_matrix(y_test, y_pred_best)
    plot_roc_curve(y_test, y_prob_mnb, y_prob_cnb, y_prob_best)
    plot_learning_curve(best_model, X_train_tfidf, y_train, "Best ComplementNB")
    plot_wordclouds(df)

    export_model(best_model, vectorizer)

    print('\n' + '=' * 60)
    print('FINAL MODEL COMPARISON')
    print('=' * 60)
    comparison = pd.DataFrame({
        'MultinomialNB': metrics_mnb,
        'ComplementNB': metrics_cnb,
        'Best ComplementNB': metrics_best
    })
    print(comparison.round(4))

    return best_model, vectorizer, metrics_best

if __name__ == "__main__":
    # Adjust filenames to match what's actually inside your extracted Kaggle download
    best_model, vectorizer, metrics = main_pipeline(
        fake_path='data/raw/Fake.csv',
        true_path='data/raw/True.csv'
    )
```

---

## 7. Future Roadmap & Phase 2+

### Phase 2: Enhanced Detection (Post-Hackathon)
- **Source Credibility Scoring:** Integrate domain reputation (NewsGuard, Media Bias/Fact Check)
- **Stylometric Features:** Readability scores, sentiment polarity, writing complexity
- **Temporal Analysis:** Detect coordinated campaigns via publication time patterns
- **Ensemble Models:** Combine Naive Bayes with Random Forest and Gradient Boosting
- **Cross-Dataset Validation:** Test on LIAR, FakeNewsNet, and WELFake datasets

### Phase 3: Production System
- **REST API:** FastAPI-based inference endpoint with <100ms latency
- **Batch Processing:** Apache Spark for large-scale article screening
- **Real-Time Stream:** Kafka integration for live news feed monitoring
- **A/B Testing:** Compare Naive Bayes vs. BERT in production environment

### Phase 4: Advanced Intelligence
- **Network Analysis:** Track how fake news spreads through social graphs
- **Multi-Modal:** Combine text with image forensics (deepfake detection)
- **Adversarial Defense:** Detect articles specifically crafted to evade detection
- **Global Expansion:** Support for Hindi, Spanish, Arabic fake news detection

---

## 8. Team Structure & RACI Matrix

### 8.1 Team Beta (4 Members)

| Role | Responsibility | Member Count |
|------|-----------------|---------------|
| Team Lead / Data Scientist | Architecture, evaluation strategy, presentation | 1 |
| ML Engineer | Model training, hyperparameter tuning, feature engineering | 1 |
| Data Engineer | Dataset acquisition, merging, EDA, preprocessing | 1 |
| Software Engineer | Visualization, word clouds, export, documentation | 1 |

### 8.2 RACI Matrix

| Task | Lead | ML Eng | Data Eng | SW Eng |
|------|------|--------|----------|--------|
| Kaggle Dataset Download | A | I | R | I |
| Dataset Loading & Merge | A | C | R | I |
| EDA & Data Validation | A | C | R | C |
| Title+Body Feature Merge | I | R | C | C |
| Advanced Preprocessing | C | R | C | I |
| TF-IDF Vectorization | I | R | C | C |
| Model Training | A | R | I | C |
| GridSearchCV Tuning | C | R | I | C |
| Evaluation Metrics | R | C | I | C |
| Confusion Matrix & ROC | I | C | I | R |
| Word Cloud Generation | I | C | I | R |
| Learning Curve | I | C | I | R |
| Model Export | I | C | I | R |
| Documentation | R | C | C | C |
| Presentation | R | C | C | C |

*R = Responsible, A = Accountable, C = Consulted, I = Informed*

---

## 9. Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|----------------------|
| Kaggle auth/download blocked (no API token, network restrictions) | Medium | High | Set up `kaggle.json` in advance; keep a manual-download fallback ready |
| Dataset column names differ from expected schema | Medium | Medium | Inspect `df.columns` immediately after loading; adjust code before running the full pipeline |
| Dataset contains political bias | High | Medium | Document bias; test on neutral topics; diversify sources in future phases |
| Class imbalance in merged data | Low | High | Check `value_counts()` before split; use ComplementNB |
| TF-IDF matrix too large for memory | Low | High | Reduce `max_features`; increase `min_df`; use `HashingVectorizer` |
| Overfitting due to title leakage | Medium | High | Ensure title is genuinely part of article content; check for exact duplicates |
| WordCloud generation timeout | Low | Low | Limit text length for word cloud; use a subset of data |
| GridSearchCV slow on large data | Medium | Medium | Use `n_jobs=-1`; reduce alpha search space; consider `RandomizedSearchCV` |

---

## 10. Appendices

### Appendix A: Glossary
- **TF-IDF:** Term Frequency-Inverse Document Frequency — weights terms by importance
- **Lemmatization:** Reducing words to dictionary root forms ("running" → "run")
- **ComplementNB:** Naive Bayes variant robust to class imbalance
- **Stratified Split:** Maintains class proportions across train/test sets
- **Word Cloud:** Visual representation of word frequency in a text corpus
- **Bigram:** Two consecutive words treated as a single feature ("fake news")

### Appendix B: Dataset Source

**Fake News Detection Datasets** — Kaggle (emineyetm)
- URL: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets
- Derived from the IEEE fake-news corpus (Reuters real articles + Politifact-flagged fake articles)
- Requires a free Kaggle account and API token for programmatic download
- **Action item for Team Beta:** confirm the exact file names and row counts once downloaded, since Kaggle mirrors of this corpus are sometimes repackaged slightly differently than the original IEEE release — update `load_and_merge_datasets()` paths accordingly

### Appendix C: References
1. Ahmed, H., Traore, I., & Saad, S. (2017). *Detection of Online Fake News Using N-Gram Analysis and Machine Learning*
2. Rennie, J. D., et al. (2003). *Tackling the Poor Assumptions of Naive Bayes Text Classifiers*
3. Scikit-learn Documentation: https://scikit-learn.org/stable/modules/naive_bayes.html
4. Shu, K., et al. (2017). *Fake News Detection on Social Media: A Data Mining Perspective*
5. Vosoughi, S., et al. (2018). *The Spread of True and False News Online* (Science)

### Appendix D: Troubleshooting Guide

**Issue:** Kaggle API returns 403/401 error
**Fix:** Regenerate `kaggle.json` from Kaggle Account settings; verify file permissions (`chmod 600 ~/.kaggle/kaggle.json`)

**Issue:** MemoryError during TF-IDF vectorization
**Fix:** Reduce `max_features` to 5000; increase `min_df` to 5; use `HashingVectorizer` instead

**Issue:** WordCloud shows only common words ("said", "will", "one")
**Fix:** Increase stopword coverage; use a custom stopword list; set `max_words=100`

**Issue:** Very long preprocessing time
**Fix:** Use pandas `.apply()` with `tqdm` for progress; sample data for testing; use multiprocessing

**Issue:** Class imbalance after merge
**Fix:** Check `value_counts()`; use ComplementNB; apply SMOTE if needed (future phase)

---

**Document End**
*Prepared by Core AI/ML Engineering Division — August 2026*
