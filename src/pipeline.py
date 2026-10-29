"""
Unified Machine Learning Pipeline for Fake News Detection
Supports Clean and Adversarially Noisy Corpora, Hyperparameter Grid Search,
Probability Calibration, and Publication Diagnostics.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc,
                             roc_auc_score)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.preprocessing import clean_text_advanced, clean_batch

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 10000
NGRAM_RANGE = (1, 2)
CV_FOLDS = 5


def load_and_merge_datasets(fake_path: str, true_path: str, drop_duplicates: bool = True) -> pd.DataFrame:
    """Load Fake.csv and True.csv, assign binary labels, and merge with deduplication."""
    print("=" * 60)
    print("STEP 1: DATASET INGESTION & AUDITING")
    print("=" * 60)
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df['label'] = 1  # Fake
    true_df['label'] = 0  # Real

    df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Total Combined Records: {df.shape[0]}")
    print(f"Class Breakdown: Fake (label=1)={int((df['label'] == 1).sum())} | Real (label=0)={int((df['label'] == 0).sum())}")

    dup_count = df.duplicated(subset=['title', 'text']).sum()
    print(f"Duplicate News Items: {dup_count}")
    if drop_duplicates and dup_count > 0:
        df = df.drop_duplicates(subset=['title', 'text']).reset_index(drop=True)
        print(f"Deduplicated dataset: {len(df)} unique records remain.")

    return df


def combine_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combines headline title and article body text into a unified representation."""
    df = df.copy()
    df['title_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    avg_len = df['title_text'].str.len().mean()
    print(f"\nFeature Space Combined: Mean text length = {avg_len:.1f} characters.")
    return df


def preprocess_dataframe(df: pd.DataFrame, text_col: str = 'title_text', n_workers: int = 4) -> pd.DataFrame:
    """
    Cleans unstructured text using parallel batch processing.
    Strips publisher datelines (Reuters/AP) to prevent data leakage.
    """
    df = df.copy()
    raw_texts = df[text_col].tolist()
    total_samples = len(raw_texts)
    print(f"\nSTEP 2: PREPROCESSING {total_samples} ARTICLES (Dateline sanitization & lemmatization)...")
    start = time.perf_counter()

    batch_size = max(500, total_samples // (n_workers * 8))
    batches = [raw_texts[i:i + batch_size] for i in range(0, total_samples, batch_size)]

    cleaned_results = []
    # Use ProcessPoolExecutor on available CPU cores
    if n_workers > 1 and total_samples > 1000:
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for batch_clean in executor.map(clean_batch, batches):
                    cleaned_results.extend(batch_clean)
        except Exception:
            # Fallback to single thread if multiprocessing is restricted
            cleaned_results = [clean_text_advanced(t, remove_datelines=True) for t in raw_texts]
    else:
        cleaned_results = [clean_text_advanced(t, remove_datelines=True) for t in raw_texts]

    df['cleaned_text'] = cleaned_results
    # Filter out empty records
    df = df[df['cleaned_text'].str.strip().str.len() > 0].reset_index(drop=True)
    elapsed = time.perf_counter() - start
    print(f"Preprocessing completed in {elapsed:.2f}s. {len(df)} usable samples retained.")
    return df


def split_data(df: pd.DataFrame, text_col: str = 'cleaned_text', label_col: str = 'label'):
    """Performs stratified train/test split to preserve exact class balance."""
    X = df[text_col]
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nSTEP 3: STRATIFIED SPLIT -> Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test


def vectorize_text(X_train, X_test):
    """Converts cleaned natural language text into sublinear TF-IDF features."""
    print("\nSTEP 4: TF-IDF VECTORIZATION (Sublinear TF, n-grams (1, 2))...")
    vectorizer = TfidfVectorizer(
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
        min_df=2,
        max_df=0.90,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Vocabulary Size: {len(vectorizer.vocabulary_)} features")
    return vectorizer, X_train_tfidf, X_test_tfidf


def train_and_tune_models(X_train_tfidf, y_train, calibrate: bool = True):
    """
    Trains baseline MultinomialNB, default ComplementNB, optimizes alpha via GridSearchCV,
    and applies probability calibration via CalibratedClassifierCV.
    """
    print("\n" + "=" * 60)
    print("STEP 5: MODEL TRAINING & HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    # 1. Baseline: MultinomialNB
    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_train_tfidf, y_train)
    mnb_cv = cross_val_score(mnb, X_train_tfidf, y_train, cv=CV_FOLDS, scoring='f1_macro')
    print(f"[Baseline] MultinomialNB (alpha=1.0) 5-Fold CV F1: {mnb_cv.mean():.4f} (+/- {mnb_cv.std():.4f})")

    # 2. Primary: Default ComplementNB
    cnb = ComplementNB(alpha=1.0)
    cnb.fit(X_train_tfidf, y_train)
    cnb_cv = cross_val_score(cnb, X_train_tfidf, y_train, cv=CV_FOLDS, scoring='f1_macro')
    print(f"[Primary]  ComplementNB (alpha=1.0) 5-Fold CV F1:  {cnb_cv.mean():.4f} (+/- {cnb_cv.std():.4f})")

    # 3. Hyperparameter Tuning
    print("Executing 5-Fold GridSearchCV over alpha parameter grid...")
    param_grid = {'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]}
    grid = GridSearchCV(ComplementNB(), param_grid, cv=CV_FOLDS, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train_tfidf, y_train)
    best_raw_model = grid.best_estimator_
    print(f"Best Tuning Alpha: {grid.best_params_['alpha']} (CV F1: {grid.best_score_:.4f})")

    # 4. Probability Calibration
    if calibrate:
        print("Calibrating posterior probability distributions (CalibratedClassifierCV)...")
        calibrated_model = CalibratedClassifierCV(estimator=best_raw_model, method='sigmoid', cv=CV_FOLDS)
        calibrated_model.fit(X_train_tfidf, y_train)
        final_model = calibrated_model
    else:
        final_model = best_raw_model

    return mnb, cnb, final_model, grid


def evaluate_model(model, X_test_tfidf, y_test, model_name: str, class_names=['Real', 'Fake']):
    """Computes comprehensive evaluation metrics on unseen holdout test split."""
    y_pred = model.predict(X_test_tfidf)
    y_prob = model.predict_proba(X_test_tfidf)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n" + "-" * 60)
    print(f"EVALUATION: {model_name}")
    print("-" * 60)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Macro:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    return y_pred, y_prob, {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc_auc
    }


def plot_diagnostics(y_test, y_pred_best, y_prob_mnb, y_prob_cnb, y_prob_best,
                     best_model, X_train_tfidf, y_train, df, prefix: str = ""):
    """Generates and saves publication-grade visual artifacts."""
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    cm_path = out_dir / f"{prefix}confusion_matrix.png"
    roc_path = out_dir / f"{prefix}roc_curve.png"
    lc_path = out_dir / f"{prefix}learning_curve.png"
    wc_fake_path = out_dir / f"{prefix}wordcloud_fake.png"
    wc_real_path = out_dir / f"{prefix}wordcloud_real.png"

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title(f'Confusion Matrix {prefix.replace("_", " ").title()}', fontsize=13, fontweight='bold')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. ROC Curves
    plt.figure(figsize=(8, 6))
    for y_prob, name, color in [
        (y_prob_mnb, 'MultinomialNB (Baseline)', 'orange'),
        (y_prob_cnb, 'ComplementNB (Default)', 'green'),
        (y_prob_best, 'Best Calibrated ComplementNB', 'blue')
    ]:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve Comparison {prefix.replace("_", " ").title()}', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Learning Curve
    train_sizes, train_scores, test_scores = learning_curve(
        best_model, X_train_tfidf, y_train, cv=CV_FOLDS, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8), scoring='f1_macro'
    )
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='blue', label='Training F1')
    plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', color='green', label='Cross-Validation F1')
    plt.xlabel('Training Samples')
    plt.ylabel('F1-Macro Score')
    plt.title(f'Learning Curve Convergence {prefix.replace("_", " ").title()}', fontsize=13, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(lc_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Word Clouds
    fake_corpus = ' '.join(df[df['label'] == 1]['cleaned_text'].dropna().head(2000))
    real_corpus = ' '.join(df[df['label'] == 0]['cleaned_text'].dropna().head(2000))

    wc_fake = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(fake_corpus)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc_fake, interpolation='bilinear')
    plt.axis('off')
    plt.title('Salient Vocabulary - Fabricated Articles', fontsize=14, color='darkred', fontweight='bold')
    plt.tight_layout()
    plt.savefig(wc_fake_path, dpi=300, bbox_inches='tight')
    plt.close()

    wc_real = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(real_corpus)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc_real, interpolation='bilinear')
    plt.axis('off')
    plt.title('Salient Vocabulary - Authentic Articles (Sanitized)', fontsize=14, color='darkblue', fontweight='bold')
    plt.tight_layout()
    plt.savefig(wc_real_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nVisual diagnostics successfully generated in {out_dir}")


def export_pipeline_assets(model, vectorizer, prefix: str = ""):
    """Exports trained and calibrated model and vectorizer artifacts."""
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{prefix}complement_nb.pkl"
    vec_path = models_dir / f"{prefix}tfidf_vectorizer.pkl"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"Model exported: {model_path}")
    print(f"Vectorizer exported: {vec_path}")


def run_pipeline(mode: str = "clean", n_workers: int = 4, calibrate: bool = True):
    """Full end-to-end execution orchestrator."""
    if mode == "clean":
        fake_csv = PROJECT_ROOT / 'data' / 'raw' / 'Fake.csv'
        true_csv = PROJECT_ROOT / 'data' / 'raw' / 'True.csv'
        prefix = ""
    else:
        fake_csv = PROJECT_ROOT / 'data' / 'processed' / 'Noisy_Fake.csv'
        true_csv = PROJECT_ROOT / 'data' / 'processed' / 'Noisy_True.csv'
        prefix = "noisy_"

    df = load_and_merge_datasets(str(fake_csv), str(true_csv), drop_duplicates=True)
    df = combine_features(df)
    df = preprocess_dataframe(df, n_workers=n_workers)

    X_train, X_test, y_train, y_test = split_data(df)
    vectorizer, X_train_tfidf, X_test_tfidf = vectorize_text(X_train, X_test)

    mnb, cnb, best_model, grid = train_and_tune_models(X_train_tfidf, y_train, calibrate=calibrate)

    y_pred_mnb, y_prob_mnb, metrics_mnb = evaluate_model(mnb, X_test_tfidf, y_test, "MultinomialNB (Baseline)")
    y_pred_cnb, y_prob_cnb, metrics_cnb = evaluate_model(cnb, X_test_tfidf, y_test, "ComplementNB (Default)")
    y_pred_best, y_prob_best, metrics_best = evaluate_model(best_model, X_test_tfidf, y_test, "Best Calibrated ComplementNB")

    plot_diagnostics(
        y_test=y_test,
        y_pred_best=y_pred_best,
        y_prob_mnb=y_prob_mnb,
        y_prob_cnb=y_prob_cnb,
        y_prob_best=y_prob_best,
        best_model=best_model,
        X_train_tfidf=X_train_tfidf,
        y_train=y_train,
        df=df,
        prefix=prefix
    )

    export_pipeline_assets(best_model, vectorizer, prefix=prefix)

    print("\n" + "=" * 60)
    print(f"FINAL METRIC MATRIX ({mode.upper()} DATASET)")
    print("=" * 60)
    summary_df = pd.DataFrame({
        'MultinomialNB': metrics_mnb,
        'ComplementNB': metrics_cnb,
        'Best Calibrated CNB': metrics_best
    }).T
    print(summary_df.round(4))
    return best_model, vectorizer, summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Fake News Machine Learning Pipeline")
    parser.add_argument("--mode", choices=["clean", "noisy", "both"], default="clean",
                        help="Choose training corpus: clean, noisy, or both")
    parser.add_argument("--workers", type=int, default=4, help="Parallel preprocessing workers")
    parser.add_argument("--no-calibration", action="store_true", help="Disable CalibratedClassifierCV")

    args = parser.parse_args()
    calibrate = not args.no_calibration

    if args.mode in ["clean", "both"]:
        print("\n>>> EXECUTING CLEAN CORPUS PIPELINE >>>")
        run_pipeline(mode="clean", n_workers=args.workers, calibrate=calibrate)

    if args.mode in ["noisy", "both"]:
        print("\n>>> EXECUTING NOISY CORPUS PIPELINE >>>")
        run_pipeline(mode="noisy", n_workers=args.workers, calibrate=calibrate)

