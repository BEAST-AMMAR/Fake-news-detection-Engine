"""
Dataset Download & Ingestion Utility
Facilitates automated downloading and verification of the ISOT Fake News Dataset
from Kaggle for training and benchmarking.
"""

import sys
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATASET_SLUG = "emineyetm/fake-news-detection-datasets"


def verify_dataset_presence() -> bool:
    """Checks whether Fake.csv and True.csv exist in data/raw."""
    fake_csv = DATA_DIR / "Fake.csv"
    true_csv = DATA_DIR / "True.csv"
    return fake_csv.exists() and true_csv.exists()


def download_dataset():
    """Attempts to download the dataset using Kaggle CLI or provides manual instructions."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if verify_dataset_presence():
        print(f"Dataset already verified in {DATA_DIR}.")
        print(f"  * Fake.csv: {(DATA_DIR / 'Fake.csv').stat().st_size / (1024*1024):.1f} MB")
        print(f"  * True.csv: {(DATA_DIR / 'True.csv').stat().st_size / (1024*1024):.1f} MB")
        return

    print("Checking Kaggle CLI availability...")
    if shutil.which("kaggle"):
        print(f"Downloading {DATASET_SLUG} via Kaggle API...")
        try:
            import subprocess
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", str(DATA_DIR), "--unzip"],
                check=True
            )
            print("Download completed successfully.")
            return
        except Exception as e:
            print(f"Automated download failed: {e}")

    print("\n" + "=" * 65)
    print("MANUAL DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 65)
    print("1. Download the dataset archive from Kaggle:")
    print("   https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets")
    print("2. Extract 'Fake.csv' and 'True.csv' directly into:")
    print(f"   {DATA_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    download_dataset()
