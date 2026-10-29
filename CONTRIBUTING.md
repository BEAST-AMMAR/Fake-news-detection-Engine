# Contributing to Fake News Detection Engine

Thank you for your interest in contributing to Project Beta! We welcome contributions that improve model accuracy, optimize inference latency, expand domain catalogs, or strengthen test coverage.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BEAST-AMMAR/Fake-news-detection-Engine.git
   cd Fake-news-detection-Engine
   ```

2. **Initialize a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Fetch required NLTK corpora:**
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
   ```

## Development Workflow

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Run tests before committing:**
   All 25 automated unit and integration tests must pass with 0 errors:
   ```bash
   pytest tests/ -v
   ```

3. **Code Style & Guidelines:**
   - Preserve type annotations across public API and inference signatures.
   - Any new text cleaning logic must be added to `src/common/preprocessing.py` to maintain training/inference parity.
   - Any new outbound network requests must include SSRF checks and timeouts.

4. **Submit a Pull Request:**
   Provide a concise description of your changes, reference any related issues, and attach relevant benchmark outputs or test results.
