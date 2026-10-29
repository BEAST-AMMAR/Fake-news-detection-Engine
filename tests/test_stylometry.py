import pytest
from src.inference.stylometry import analyze_heuristics, analyze_stylometry


def test_analyze_heuristics_calm_text():
    calm_title = "Treasury Secretary announces new tax compliance guidelines"
    calm_body = "The Treasury Department issued formal guidelines on corporate reporting standards."
    res = analyze_heuristics(calm_body, title=calm_title)
    assert res["heuristic_suspicion_score"] <= 0.1
    assert len(res["sensational_hits"]) == 0


def test_analyze_heuristics_detects_headline_clickbait():
    sensational_title = "SHOCKING BOMBSHELL: YOU WON'T BELIEVE WHAT DEEP STATE DID!"
    calm_body = "An internal document was discussed during the committee hearing."
    res = analyze_heuristics(calm_body, title=sensational_title)
    # The sensational title with triggers and high caps must produce a noticeable suspicion score
    assert res["heuristic_suspicion_score"] >= 0.4
    assert len(res["sensational_hits"]) >= 1


def test_analyze_stylometry_returns_valid_probabilities():
    text = "The committee reviewed the economic reports submitted by financial institutions."
    res = analyze_stylometry(text, title="Economic Committee Review")
    assert "fake_probability" in res
    assert "real_probability" in res
    assert 0.0 <= res["fake_probability"] <= 1.0
    assert 0.0 <= res["real_probability"] <= 1.0
    assert round(res["fake_probability"] + res["real_probability"], 2) == 1.0

