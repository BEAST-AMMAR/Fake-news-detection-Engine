"""
Project Beta: Fake News Detection - Noisy Corpus Benchmark Pipeline Entrypoint
Maintains full backward compatibility while routing to the unified src.pipeline engine.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (
    load_and_merge_datasets,
    combine_features,
    preprocess_dataframe,
    split_data,
    vectorize_text,
    train_and_tune_models,
    evaluate_model,
    export_pipeline_assets,
    run_pipeline
)

def noisy_pipeline(fake_path=None, true_path=None):
    """Legacy entrypoint preserved for backward compatibility."""
    model, vectorizer, summary = run_pipeline(mode="noisy", n_workers=4, calibrate=True)
    return model, vectorizer, summary

if __name__ == "__main__":
    noisy_pipeline()
