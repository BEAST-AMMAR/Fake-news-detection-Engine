import pytest
from src.common.preprocessing import (
    strip_publisher_datelines,
    clean_text_advanced,
    ALLOWED_SHORT_TOKENS
)


def test_strip_publisher_datelines():
    # Reuters datelines
    reuters_text = "WASHINGTON (Reuters) - The Federal Reserve maintained interest rates on Wednesday."
    assert "reuters" not in strip_publisher_datelines(reuters_text).lower()
    assert strip_publisher_datelines(reuters_text).startswith("The Federal Reserve")

    # AP datelines
    ap_text = "LONDON (AP) — British lawmakers gathered for an emergency vote."
    assert "ap" not in strip_publisher_datelines(ap_text).lower()
    assert strip_publisher_datelines(ap_text).startswith("British lawmakers")

    # Combined locations
    multi_loc = "BEIJING/GENEVA (Reuters) - Diplomatic negotiations commenced."
    assert strip_publisher_datelines(multi_loc).startswith("Diplomatic negotiations")


def test_clean_text_advanced_preserves_acronyms():
    sample = "The UN and EU agreed on AI safety policies for the US and UK."
    cleaned = clean_text_advanced(sample, remove_datelines=True)
    tokens = cleaned.split()
    assert "ai" in tokens
    assert "eu" in tokens
    assert "un" in tokens
    assert "usa" in tokens  # Normalized country code
    assert "uk" in tokens


def test_clean_text_advanced_removes_urls_and_html():
    sample = "<p>Check out our report at https://example.com/breaking-news today.</p>"
    cleaned = clean_text_advanced(sample)
    assert "http" not in cleaned
    assert "example" not in cleaned
    assert "<p>" not in cleaned


def test_clean_text_advanced_normalizes_standalone_numbers():
    sample = "In 2024 there were 100 delegates attending the 5G conference."
    cleaned = clean_text_advanced(sample)
    tokens = cleaned.split()
    assert "2024" not in tokens
    assert "100" not in tokens
    assert "5g" in tokens  # 5G is in ALLOWED_SHORT_TOKENS


def test_clean_text_advanced_empty_inputs():
    assert clean_text_advanced("") == ""
    assert clean_text_advanced(None) == ""
    assert clean_text_advanced("    ") == ""
