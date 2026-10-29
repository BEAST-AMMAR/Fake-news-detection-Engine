"""
Real-Time Tribrid Fake News Detection Engine
Combines:
1. Domain Reputation & Source Authenticity
2. Stylometric NLP & Naive Bayes Classifier
3. Live Fact-Checking Consensus & Wire Cross-Referencing
"""

from .fusion_engine import TribridEngine, VerificationResult

__all__ = ["TribridEngine", "VerificationResult"]
