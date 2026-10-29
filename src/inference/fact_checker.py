"""
Pillar 3: Live Fact Consensus & Wire Cross-Referencing
Queries Google Fact Check Tools API (ClaimReview standard) and performs
live multi-wire news search cross-referencing via Google News RSS feeds.
Hardened with bidirectional semantic overlap calculations and memory TTL caching.
"""

import os
import re
import html
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional
import httpx


GOOGLE_FACT_CHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

ACCREDITED_NEWS_SOURCES = {
    "npr", "reuters", "associated press", "ap news", "ap", "bbc", "bbc news",
    "the guardian", "the new york times", "new york times", "the wall street journal",
    "wall street journal", "wsj", "bloomberg", "the washington post", "washington post",
    "al jazeera", "deutsche welle", "france 24", "cnn", "nbc news", "cbs news", "abc news",
    "time", "nature", "science", "the hindu", "indian express", "ndtv", "iol", "news24", "afp"
}

QUERY_STOP_WORDS = {
    "the", "a", "an", "in", "on", "at", "of", "for", "to", "and", "or", "by",
    "that", "this", "with", "from", "as", "is", "are", "was", "were", "be",
    "has", "have", "had", "will", "would", "shall", "should", "it", "its",
    "said", "says", "told", "reports", "reported", "breaking", "news",
    "today", "yesterday", "statement", "officials", "official", "update"
}

# In-memory TTL caches to avoid redundant API/RSS network latency
_FACT_CHECK_CACHE: Dict[str, tuple[float, List[Dict[str, Any]]]] = {}
_WIRE_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}
CACHE_TTL = 600  # 10 minutes


def get_token_keywords(text: str) -> set:
    """Extracts distinctive content tokens for semantic overlap matching."""
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    return {w for w in words if w not in QUERY_STOP_WORDS and len(w) > 2}


def compute_keyword_overlap(source_tokens: set, candidate_tokens: set) -> float:
    """
    Computes weighted bidirectional similarity between query and candidate headline.
    Blends directional recall with Jaccard index to prevent false corroboration
    on queries that merely share common context words.
    """
    if not source_tokens or not candidate_tokens:
        return 0.0

    intersection = source_tokens.intersection(candidate_tokens)
    if len(intersection) < 2:
        return 0.0

    union = source_tokens.union(candidate_tokens)
    jaccard = len(intersection) / len(union)
    directional_recall = len(intersection) / len(source_tokens)

    # Harmonic blend favoring tight Jaccard alignment
    combined_score = (0.35 * directional_recall) + (0.65 * jaccard)
    return round(combined_score, 3)


def extract_searchable_query(title: str, text: str, max_words: int = 14) -> str:
    """Extracts a succinct, search-friendly query from title or lead sentence."""
    target = title.strip() if title and len(title.strip()) > 8 else text.strip()
    cleaned = re.sub(r'["\']', '', target)
    words = cleaned.split()[:max_words]
    return " ".join(words)


async def query_google_fact_check(query: str, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Queries Google Fact Check Tools API for existing ClaimReview debunks with caching.
    """
    normalized_q = query.strip().lower()
    now = time.time()
    if normalized_q in _FACT_CHECK_CACHE:
        cached_time, cached_res = _FACT_CHECK_CACHE[normalized_q]
        if now - cached_time < CACHE_TTL:
            return cached_res

    key = api_key or os.getenv("GOOGLE_FACTCHECK_API_KEY")
    if not key or not query:
        return []

    params = {
        "query": query,
        "key": key,
        "languageCode": "en-US",
        "pageSize": 5
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(GOOGLE_FACT_CHECK_API_URL, params=params)
            if resp.status_code == 200:
                data = resp.json()
                claims = data.get("claims", [])
                results = []
                for claim in claims:
                    text = claim.get("text", "")
                    claimant = claim.get("claimant", "Unknown")
                    reviews = claim.get("claimReview", [])
                    for rev in reviews:
                        publisher = rev.get("publisher", {}).get("name", "Fact Checker")
                        rating = rev.get("textualRating", "Unrated")
                        url = rev.get("url", "")
                        results.append({
                            "claim": text,
                            "claimant": claimant,
                            "publisher": publisher,
                            "rating": rating,
                            "url": url,
                            "source": "Google Fact Check Tools API"
                        })
                _FACT_CHECK_CACHE[normalized_q] = (now, results)
                return results
    except Exception:
        pass
    return []


async def query_wire_consensus(query: str) -> Dict[str, Any]:
    """
    Searches Google News RSS feeds for real-time journalistic coverage
    and measures semantic overlap against accredited wire publishers with caching.
    """
    if not query:
        return {
            "wire_count": 0,
            "reuters_matched": False,
            "ap_matched": False,
            "bbc_matched": False,
            "snippets": [],
            "reputable_sources": [],
            "max_overlap": 0.0,
            "search_performed": False
        }

    normalized_q = query.strip().lower()
    now = time.time()
    if normalized_q in _WIRE_CACHE:
        cached_time, cached_res = _WIRE_CACHE[normalized_q]
        if now - cached_time < CACHE_TTL:
            return cached_res

    q_tokens = get_token_keywords(query)
    encoded = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        )
    }

    try:
        async with httpx.AsyncClient(headers=headers, timeout=8.0, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                root = ET.fromstring(resp.text)
                items = root.findall("./channel/item")

                reputable_sources = []
                snippets = []
                max_overlap = 0.0

                reuters_matched = False
                ap_matched = False
                bbc_matched = False

                for item in items[:8]:
                    raw_title = item.find("title").text if item.find("title") is not None else ""
                    source = item.find("source").text if item.find("source") is not None else ""

                    clean_title = html.unescape(raw_title).strip()
                    t_tokens = get_token_keywords(clean_title)
                    overlap = compute_keyword_overlap(q_tokens, t_tokens)

                    if overlap > max_overlap:
                        max_overlap = overlap

                    s_lower = source.lower()
                    if "reuters" in s_lower:
                        reuters_matched = True
                    if "associated press" in s_lower or s_lower == "ap" or "ap news" in s_lower:
                        ap_matched = True
                    if "bbc" in s_lower:
                        bbc_matched = True

                    is_rep = any(rep in s_lower for rep in ACCREDITED_NEWS_SOURCES)
                    # Hardened threshold: require at least 0.45 weighted overlap to count as corroborating
                    if is_rep and overlap >= 0.45 and source not in reputable_sources:
                        reputable_sources.append(source)

                    if overlap >= 0.30 and len(snippets) < 3:
                        snippets.append(f"[{source}] {clean_title}")

                result = {
                    "wire_count": len(reputable_sources),
                    "reuters_matched": reuters_matched,
                    "ap_matched": ap_matched,
                    "bbc_matched": bbc_matched,
                    "reputable_sources": reputable_sources,
                    "snippets": snippets,
                    "max_overlap": round(max_overlap, 3),
                    "search_performed": True,
                    "note": f"Found {len(items)} indexed news items; max headline overlap {max_overlap*100:.1f}%."
                }
                _WIRE_CACHE[normalized_q] = (now, result)
                return result
    except Exception:
        pass

    fallback_res = {
        "wire_count": 0,
        "reuters_matched": False,
        "ap_matched": False,
        "bbc_matched": False,
        "snippets": [],
        "reputable_sources": [],
        "max_overlap": 0.0,
        "search_performed": False,
        "note": "News wire search temporarily unavailable."
    }
    return fallback_res


def interpret_fact_check_verdicts(debunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyzes textual ratings from fact-checking reviews (e.g. Snopes, PolitiFact)."""
    if not debunks:
        return {
            "has_debunk": False,
            "debunk_verdict": None,
            "debunk_severity": 0.0
        }

    false_indicators = ["false", "pants on fire", "incorrect", "hoax", "fake", "misleading", "fabricated", "untrue"]
    true_indicators = ["true", "correct", "verified", "accurate"]
    satire_indicators = ["satire", "parody", "humor"]

    false_hits = 0
    true_hits = 0
    satire_hits = 0

    for d in debunks:
        r = d.get("rating", "").lower()
        if any(ind in r for ind in false_indicators):
            false_hits += 1
        elif any(ind in r for ind in true_indicators):
            true_hits += 1
        elif any(ind in r for ind in satire_indicators):
            satire_hits += 1

    if satire_hits > 0:
        return {"has_debunk": True, "debunk_verdict": "SATIRE", "debunk_severity": 0.8}
    elif false_hits > 0:
        return {"has_debunk": True, "debunk_verdict": "FALSE", "debunk_severity": 1.0}
    elif true_hits > 0:
        return {"has_debunk": True, "debunk_verdict": "VERIFIED_TRUE", "debunk_severity": 0.0}

    return {"has_debunk": True, "debunk_verdict": "INCONCLUSIVE", "debunk_severity": 0.5}


async def check_facts(title: str, text: str) -> Dict[str, Any]:
    """
    Primary entrypoint for Pillar 3:
    1. Extracts query from title or text
    2. Runs Google Fact Check search (ClaimReview)
    3. Runs real-time news wire RSS consensus search
    """
    query = extract_searchable_query(title, text)
    if not query:
        return {
            "query": "",
            "fact_checks": [],
            "debunk_summary": {"has_debunk": False, "debunk_verdict": None},
            "wire_consensus": {"wire_count": 0, "search_performed": False},
            "consensus_score": 0.5,
            "status": "No sufficient text or title to search facts."
        }

    # Concurrently execute fact checks & wire searches
    debunks = await query_google_fact_check(query)
    wires = await query_wire_consensus(query)
    debunk_summary = interpret_fact_check_verdicts(debunks)

    # Calculate Consensus Score:
    # 1. Direct Debunks take highest priority
    if debunk_summary["has_debunk"]:
        if debunk_summary["debunk_verdict"] == "FALSE":
            consensus_score = 0.05
        elif debunk_summary["debunk_verdict"] == "SATIRE":
            consensus_score = 0.10
        elif debunk_summary["debunk_verdict"] == "VERIFIED_TRUE":
            consensus_score = 0.95
        else:
            consensus_score = 0.40
    elif wires.get("search_performed"):
        wire_cnt = wires.get("wire_count", 0)
        max_overlap = wires.get("max_overlap", 0.0)

        # Corroborated by accredited wire with high bidirectional similarity
        if wire_cnt >= 1 and max_overlap >= 0.45:
            consensus_score = 0.95
        elif max_overlap >= 0.60:
            consensus_score = 0.90
        elif max_overlap >= 0.35:
            consensus_score = 0.70
        elif max_overlap < 0.20:
            # Low wire overlap - neutral baseline (absence of evidence != proof of hoax)
            consensus_score = 0.40
        else:
            consensus_score = 0.50
    else:
        # Search unreachable / offline
        consensus_score = 0.50

    return {
        "query": query,
        "fact_checks": debunks,
        "debunk_summary": debunk_summary,
        "wire_consensus": wires,
        "consensus_score": round(consensus_score, 3),
        "status": "Fact-checking consensus completed."
    }
