"""
Pillar 1: Domain Reputation & Source Authenticity Verifier
Inspects publisher domain using registered domain catalogs (reputable wires,
fact-checking institutions, satire outlets, and known disinformation domains).
"""

from typing import Dict, Any, Optional
import tldextract


# Curated lists grounded in Media Bias/Fact Check (MBFC) and OpenSources classifications
REPUTABLE_OUTLETS = {
    # News Wires & Public Broadcasters
    "reuters.com": "Global news wire with high factual reporting",
    "apnews.com": "Associated Press - Primary news wire with strict verification",
    "bbc.com": "BBC News - Public broadcaster with editorial guidelines",
    "bbc.co.uk": "BBC News UK - Public service journalism",
    "afp.com": "Agence France-Presse - Primary international wire",
    "npr.org": "National Public Radio - High factual reporting",
    "pbs.org": "PBS NewsHour - Trusted public broadcasting",
    
    # Established International & National Press
    "theguardian.com": "The Guardian - High factual reporting",
    "nytimes.com": "The New York Times - Established newspaper of record",
    "wsj.com": "The Wall Street Journal - Top business & news publication",
    "bloomberg.com": "Bloomberg News - Financial and global news wire",
    "washingtonpost.com": "The Washington Post - Established daily reporting",
    "ft.com": "Financial Times - High editorial standards",
    "economist.com": "The Economist - Rigorous analytical reporting",
    "time.com": "Time Magazine - Longform journalism",
    "nature.com": "Nature - Top peer-reviewed scientific journal",
    "sciencemag.org": "Science Magazine - Peer-reviewed scientific authority",
    "aljazeera.com": "Al Jazeera English - Major international broadcaster",
    "dw.com": "Deutsche Welle - German public international broadcaster",
    "france24.com": "France 24 - French state-owned international news",
    "thehindu.com": "The Hindu - National newspaper of record in India",
    "indianexpress.com": "The Indian Express - Established investigative journalism",
    "ndtv.com": "NDTV - Established news network",
}

FACT_CHECKERS = {
    "snopes.com": "Snopes - Renowned independent fact-checking organization",
    "politifact.com": "PolitiFact - Pulitzer Prize-winning fact-checker",
    "factcheck.org": "FactCheck.org - Annenberg Public Policy Center project",
    "fullfact.org": "Full Fact - UK independent fact-checking charity",
    "leadstories.com": "Lead Stories - Certified fact-checking organization",
    "checkyourfact.com": "Check Your Fact - IFCN certified fact-checker",
    "boomlive.in": "BOOM Live - Certified fact-checking initiative in South Asia",
    "altnews.in": "Alt News - Dedicated disinformation fact-checking portal",
    "vishvasnews.com": "Vishvas News - Certified fact-checking desk",
}

SATIRE_OUTLETS = {
    "theonion.com": "The Onion - Famous satirical news organization",
    "babylonbee.com": "The Babylon Bee - Christian and political satire publication",
    "clickhole.com": "ClickHole - Parody of internet clickbait culture",
    "newsthump.com": "NewsThump - UK-based satirical news service",
    "thedailymash.co.uk": "The Daily Mash - British satirical website",
    "waterfordwhispersnews.com": "Waterford Whispers News - Irish satirical website",
    "borowitzreport.com": "The Borowitz Report - Satirical column",
    "fakingnews.com": "Faking News - Indian satire portal",
}

DISINFO_OUTLETS = {
    "infowars.com": "Known conspiracy theory and fabricated news site",
    "naturalnews.com": "Pseudoscience and medical conspiracy website",
    "beforeitsnews.com": "Unvetted user-submitted conspiracy blog",
    "yournewswire.com": "Frequent publisher of hoaxes and fake stories",
    "newspunch.com": "Rebranded outlet with history of manufactured stories",
    "worldnewsdailyreport.com": "Fabricated hoax and shock-value articles",
    "dcgazette.com": "Known fake news clickbait generator",
    "thegatewaypundit.com": "Repeatedly flagged for misleading claims and false stories",
    "breitbart.com": "Extremely biased with frequent false and unverified reporting",
    "prntly.com": "Known hyper-partisan hoax website",
    "abcnews.com.co": "Known typo-squatting counterfeit domain impersonating ABC News",
    "cnn-trending.com": "Impersonator scam domain",
}


def extract_registered_domain(url: str) -> str:
    """Extracts the registered root domain (e.g. bbc.co.uk from https://news.bbc.co.uk/...)"""
    extracted = tldextract.extract(url)
    domain = getattr(extracted, "top_domain_under_public_suffix", None) or getattr(extracted, "registered_domain", "")
    if domain:
        return domain.lower()
    return ""


def verify_domain(url_or_domain: str) -> Dict[str, Any]:
    """
    Evaluates domain reputation against curated databases.
    Returns domain classification, trust weight, and explanatory rationale.
    """
    if not url_or_domain:
        return {
            "domain": None,
            "category": "UNKNOWN",
            "trust_score": 0.5,
            "status_label": "No domain provided (Raw text input)",
            "details": "Input is raw text without an originating source domain URL."
        }
    
    domain = extract_registered_domain(url_or_domain) if ("." in url_or_domain or "/" in url_or_domain) else url_or_domain.lower()

    if not domain:
        return {
            "domain": url_or_domain,
            "category": "INVALID",
            "trust_score": 0.5,
            "status_label": "Unparseable Domain",
            "details": f"Could not extract registered domain from '{url_or_domain}'."
        }

    # Check Registries
    if domain in REPUTABLE_OUTLETS:
        return {
            "domain": domain,
            "category": "REPUTABLE",
            "trust_score": 0.95,
            "status_label": "Accredited News Publisher",
            "details": REPUTABLE_OUTLETS[domain]
        }
    
    if domain in FACT_CHECKERS:
        return {
            "domain": domain,
            "category": "FACT_CHECKER",
            "trust_score": 0.98,
            "status_label": "Certified Fact-Checking Authority",
            "details": FACT_CHECKERS[domain]
        }

    if domain in SATIRE_OUTLETS:
        return {
            "domain": domain,
            "category": "SATIRE",
            "trust_score": 0.10,
            "status_label": "Known Satire & Parody Outlet",
            "details": SATIRE_OUTLETS[domain]
        }

    if domain in DISINFO_OUTLETS:
        return {
            "domain": domain,
            "category": "DISINFORMATION",
            "trust_score": 0.05,
            "status_label": "Flagged Disinformation & Hoax Domain",
            "details": DISINFO_OUTLETS[domain]
        }

    # Unknown / Uncategorized domain
    return {
        "domain": domain,
        "category": "UNCATEGORIZED",
        "trust_score": 0.50,
        "status_label": "Uncategorized Web Domain",
        "details": f"'{domain}' is not in the curated high-reputation or flagged blacklist databases. Relying on stylometrics and fact-check cross-referencing."
    }
