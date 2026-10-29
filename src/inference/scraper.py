"""
Scraper Module
Extracts clean article title, metadata, and body text from raw URLs using Trafilatura
with an HTTPX fallback. Hardened with SSRF protection and robust title fallbacks.
"""

import re
import socket
import ipaddress
from urllib.parse import urlparse
from typing import Dict, Any, Optional
import httpx
import trafilatura
from trafilatura.settings import use_config

# Configure Trafilatura for fast extraction
config = use_config()
config.set("DEFAULT", "EXTRACTION_TIMEOUT", "10")


def is_safe_url(url: str) -> tuple[bool, str]:
    """
    Validates URL scheme and resolves hostname to ensure it does not target
    private, loopback, or cloud-metadata IP addresses (SSRF mitigation).
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme.lower() not in ("http", "https"):
            return False, f"Unsupported URL scheme '{parsed.scheme}'. Only http and https are allowed."

        hostname = parsed.hostname
        if not hostname:
            return False, "Invalid or missing hostname in URL."

        # Quick check for literal localhost
        if hostname.lower() in ("localhost", "127.0.0.1", "::1", "metadata.google.internal"):
            return False, f"Access to restricted hostname '{hostname}' is blocked."

        # Resolve IP addresses
        addr_info = socket.getaddrinfo(hostname, None)
        for _, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False, f"Access to private/internal network address ({ip_str}) is prohibited."

        return True, ""
    except Exception as e:
        return False, f"Could not validate target URL: {str(e)}"


def extract_slug_title(url: str) -> str:
    """Extracts fallback article title from the URL slug if page scraping misses it."""
    try:
        path = urlparse(url).path.strip("/")
        if not path:
            return ""
        last_segment = path.split("/")[-1]
        # Strip file extensions
        last_segment = re.sub(r'\.(?:html?|php|aspx?)$', '', last_segment, flags=re.IGNORECASE)
        # Only treat as slug if it has hyphen/underscore separators and reasonable length
        if len(last_segment) > 5 and ("-" in last_segment or "_" in last_segment):
            words = re.split(r'[-_]+', last_segment)
            return " ".join(word.capitalize() for word in words if word)
    except Exception:
        pass
    return ""


async def scrape_article_from_url(url: str, timeout_seconds: float = 12.0) -> Dict[str, Any]:
    """
    Downloads and extracts article title, body text, author, and date.
    Returns structured dictionary with extracted content or error information.
    """
    slug_title = extract_slug_title(url)

    # SSRF Pre-flight check
    safe, err_msg = is_safe_url(url)
    if not safe:
        return {
            "success": False,
            "url": url,
            "title": slug_title,
            "text": "",
            "author": None,
            "date": None,
            "error": f"Security restriction: {err_msg}"
        }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    html_content = None
    try:
        # Step 1: Attempt fast async fetch using httpx with modern user agent
        async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True, headers=headers) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                html_content = resp.text
    except Exception:
        # Fallback to trafilatura's internal fetcher
        pass

    if not html_content:
        # Fallback synchronous fetch
        try:
            html_content = trafilatura.fetch_url(url)
        except Exception:
            html_content = None

    if not html_content:
        return {
            "success": False,
            "url": url,
            "title": slug_title,
            "text": "",
            "author": None,
            "date": None,
            "error": "Failed to fetch webpage content (Site may block crawlers, have an anti-bot paywall, or timed out)."
        }

    # Step 2: Extract structured text and metadata
    extracted_text = trafilatura.extract(
        html_content,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
        config=config
    )
    
    metadata = trafilatura.extract_metadata(html_content)

    title = slug_title
    author = None
    date = None

    if metadata:
        title = metadata.title or slug_title
        author = metadata.author
        date = metadata.date

    if not extracted_text:
        return {
            "success": False,
            "url": url,
            "title": title,
            "text": "",
            "author": author,
            "date": date,
            "error": "Webpage was fetched, but no readable article text could be extracted (may require JavaScript rendering or is an image/video post)."
        }

    return {
        "success": True,
        "url": url,
        "title": title,
        "text": extracted_text,
        "author": author,
        "date": date,
        "word_count": len(extracted_text.split()),
        "error": None
    }
