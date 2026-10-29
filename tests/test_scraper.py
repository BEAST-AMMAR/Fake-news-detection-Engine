import pytest
from src.inference.scraper import is_safe_url, extract_slug_title


def test_is_safe_url_blocks_loopback_and_private_ips():
    # Localhost
    safe, msg = is_safe_url("http://localhost:8000/admin")
    assert not safe
    assert "restricted" in msg.lower() or "blocked" in msg.lower()

    # Literal loopback
    safe, msg = is_safe_url("http://127.0.0.1:8000/api")
    assert not safe

    # Cloud metadata link-local IP
    safe, msg = is_safe_url("http://169.254.169.254/latest/meta-data/")
    assert not safe

    # RFC 1918 private network
    safe, msg = is_safe_url("http://192.168.1.1/router")
    assert not safe


def test_is_safe_url_blocks_unsupported_schemes():
    safe, msg = is_safe_url("ftp://example.com/file.txt")
    assert not safe
    assert "scheme" in msg.lower()

    safe, msg = is_safe_url("file:///etc/passwd")
    assert not safe


def test_is_safe_url_permits_legitimate_domains():
    safe, msg = is_safe_url("https://www.reuters.com/markets/macro/")
    assert safe
    assert msg == ""


def test_extract_slug_title():
    url = "https://www.example.com/news/scientists-discover-breakthrough-treatment.html"
    title = extract_slug_title(url)
    assert "Scientists Discover Breakthrough Treatment" in title

    url_simple = "https://www.example.com/no-slug"
    assert extract_slug_title(url_simple) == "No Slug"

