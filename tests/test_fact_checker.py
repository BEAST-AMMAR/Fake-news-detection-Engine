import pytest
from src.inference.fact_checker import (
    get_token_keywords,
    compute_keyword_overlap,
    extract_searchable_query,
    interpret_fact_check_verdicts
)


def test_get_token_keywords():
    text = "The Federal Reserve raised interest rates today in Washington."
    tokens = get_token_keywords(text)
    assert "federal" in tokens
    assert "reserve" in tokens
    assert "interest" in tokens
    # Stop words / generic words should be omitted
    assert "the" not in tokens
    assert "in" not in tokens
    assert "today" not in tokens


def test_compute_keyword_overlap_identical():
    tokens1 = {"federal", "reserve", "interest", "rates"}
    tokens2 = {"federal", "reserve", "interest", "rates"}
    overlap = compute_keyword_overlap(tokens1, tokens2)
    assert overlap == 1.0


def test_compute_keyword_overlap_disjoint():
    tokens1 = {"aliens", "secret", "moon", "base"}
    tokens2 = {"economy", "inflation", "market", "stocks"}
    overlap = compute_keyword_overlap(tokens1, tokens2)
    assert overlap == 0.0


def test_compute_keyword_overlap_adversarial_trick_does_not_corroborate():
    # Hoax query sharing some context words with a real news wire
    query_tokens = {"white", "house", "scientists", "discover", "secret", "aliens", "running", "government"}
    candidate_tokens = {"white", "house", "scientists", "discover", "clean", "energy", "climate", "initiative"}
    
    overlap = compute_keyword_overlap(query_tokens, candidate_tokens)
    # The weighted bidirectional metric must stay below the corroboration threshold (0.45)
    assert overlap < 0.45


def test_extract_searchable_query():
    title = "Breaking: European central bank announces emergency monetary meeting"
    query = extract_searchable_query(title, text="", max_words=6)
    assert query == "Breaking: European central bank announces emergency"


def test_interpret_fact_check_verdicts():
    false_reviews = [{"rating": "False", "claim": "Hoax claim"}]
    res = interpret_fact_check_verdicts(false_reviews)
    assert res["has_debunk"] is True
    assert res["debunk_verdict"] == "FALSE"

    satire_reviews = [{"rating": "Satire", "claim": "Humorous piece"}]
    res = interpret_fact_check_verdicts(satire_reviews)
    assert res["has_debunk"] is True
    assert res["debunk_verdict"] == "SATIRE"

    empty_reviews = []
    res = interpret_fact_check_verdicts(empty_reviews)
    assert res["has_debunk"] is False

