"""
Tribrid Inference Fusion Engine
Combines Domain Verifier, Stylometric NLP Classifier, and Live Fact Consensus
using a Smart Waterfall priority arbiter. Hardened against false corroboration
and absence-of-evidence penalties.
"""

import time
import asyncio
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from .domain_verifier import verify_domain
from .scraper import scrape_article_from_url
from .stylometry import analyze_stylometry
from .fact_checker import check_facts


class VerificationResult(BaseModel):
    url: Optional[str] = None
    title: str = ""
    author: Optional[str] = None
    publish_date: Optional[str] = None
    word_count: int = 0
    credibility_score: float = Field(..., description="0 to 100 score where 100 is highly authentic and 0 is fake")
    verdict: str = Field(..., description="Categorical verdict: VERIFIED_TRUE, LIKELY_REAL, UNVERIFIED_UNCERTAIN, LIKELY_FAKE, CONFIRMED_FAKE_OR_HOAX, SATIRE_PARODY")
    confidence_level: str = Field(..., description="HIGH, MEDIUM, LOW")
    primary_reason: str = ""
    pillar_1_domain: Dict[str, Any]
    pillar_2_stylometry: Dict[str, Any]
    pillar_3_fact_consensus: Dict[str, Any]
    execution_time_ms: float


class TribridEngine:
    """Orchestrates asynchronous Tribrid inference pipeline."""

    def __init__(self):
        pass

    async def verify(self, url: Optional[str] = None, text: Optional[str] = None, title: Optional[str] = None) -> VerificationResult:
        start_time = time.perf_counter()

        extracted_title = title or ""
        article_text = text or ""
        author = None
        publish_date = None
        domain_info = {}

        # Step 1: Scrape if URL provided
        if url:
            domain_task = asyncio.to_thread(verify_domain, url)
            scrape_task = scrape_article_from_url(url)
            domain_info, scrape_result = await asyncio.gather(domain_task, scrape_task)

            if scrape_result.get("success"):
                extracted_title = scrape_result.get("title") or extracted_title
                article_text = scrape_result.get("text") or article_text
                author = scrape_result.get("author")
                publish_date = scrape_result.get("date")
            else:
                extracted_title = scrape_result.get("title") or extracted_title
                dom_cat = domain_info.get("category")
                if dom_cat == "SATIRE":
                    elapsed = (time.perf_counter() - start_time) * 1000
                    return VerificationResult(
                        url=url,
                        title=extracted_title,
                        author=author,
                        publish_date=publish_date,
                        word_count=0,
                        credibility_score=10.0,
                        verdict="SATIRE_PARODY",
                        confidence_level="HIGH",
                        primary_reason=f"Published on confirmed satire/parody outlet ({domain_info.get('details')}). Not intended as factual news.",
                        pillar_1_domain=domain_info,
                        pillar_2_stylometry={},
                        pillar_3_fact_consensus={},
                        execution_time_ms=round(elapsed, 2)
                    )
                elif dom_cat == "DISINFORMATION":
                    elapsed = (time.perf_counter() - start_time) * 1000
                    return VerificationResult(
                        url=url,
                        title=extracted_title,
                        author=author,
                        publish_date=publish_date,
                        word_count=0,
                        credibility_score=5.0,
                        verdict="CONFIRMED_FAKE_OR_HOAX",
                        confidence_level="HIGH",
                        primary_reason=f"Published on confirmed disinformation/hoax domain ({domain_info.get('domain')}). Blacklisted by media watchdogs.",
                        pillar_1_domain=domain_info,
                        pillar_2_stylometry={},
                        pillar_3_fact_consensus={},
                        execution_time_ms=round(elapsed, 2)
                    )
                elif not article_text:
                    elapsed = (time.perf_counter() - start_time) * 1000
                    if dom_cat in ["REPUTABLE", "FACT_CHECKER"]:
                        return VerificationResult(
                            url=url,
                            title=extracted_title,
                            author=author,
                            publish_date=publish_date,
                            word_count=0,
                            credibility_score=88.0,
                            verdict="LIKELY_REAL",
                            confidence_level="MEDIUM",
                            primary_reason=f"Published by accredited news organization ({domain_info.get('domain')}). Full article body was behind subscriber paywall/anti-bot protection.",
                            pillar_1_domain=domain_info,
                            pillar_2_stylometry={},
                            pillar_3_fact_consensus={},
                            execution_time_ms=round(elapsed, 2)
                        )
                    return VerificationResult(
                        url=url,
                        title=extracted_title,
                        author=author,
                        publish_date=publish_date,
                        word_count=0,
                        credibility_score=domain_info.get("trust_score", 0.5) * 100,
                        verdict="UNVERIFIED_UNCERTAIN",
                        confidence_level="LOW",
                        primary_reason=f"Could not extract body text: {scrape_result.get('error')}",
                        pillar_1_domain=domain_info,
                        pillar_2_stylometry={},
                        pillar_3_fact_consensus={},
                        execution_time_ms=round(elapsed, 2)
                    )
        else:
            domain_info = verify_domain("")

        # Step 2: Execute Stylometry and Fact-Checking concurrently
        # Pass both title and text so that both the ML vectorizer and heuristics examine the headline
        stylometry_task = asyncio.to_thread(analyze_stylometry, article_text, title=extracted_title)
        fact_task = check_facts(extracted_title, article_text)

        stylometry_result, fact_result = await asyncio.gather(stylometry_task, fact_task)

        # Step 3: Smart Waterfall Arbitration
        credibility, verdict, confidence, reason = self._arbitrate(
            domain_info=domain_info,
            stylometry=stylometry_result,
            fact_consensus=fact_result,
            has_url=bool(url)
        )

        elapsed = (time.perf_counter() - start_time) * 1000

        return VerificationResult(
            url=url,
            title=extracted_title,
            author=author,
            publish_date=publish_date,
            word_count=len(article_text.split()),
            credibility_score=round(credibility, 1),
            verdict=verdict,
            confidence_level=confidence,
            primary_reason=reason,
            pillar_1_domain=domain_info,
            pillar_2_stylometry=stylometry_result,
            pillar_3_fact_consensus=fact_result,
            execution_time_ms=round(elapsed, 2)
        )

    def _arbitrate(
        self,
        domain_info: Dict[str, Any],
        stylometry: Dict[str, Any],
        fact_consensus: Dict[str, Any],
        has_url: bool
    ) -> tuple[float, str, str, str]:
        """
        Waterfall rules:
        1. Satire Outlet -> SATIRE_PARODY (12%)
        2. Known Disinfo Domain -> CONFIRMED_FAKE_OR_HOAX (6%)
        3. Direct Fact-Check Debunk -> CONFIRMED_FAKE_OR_HOAX (8%)
        4. Direct Fact-Check Verified -> VERIFIED_TRUE (94%)
        5. Accredited News Wire + Wire Signals -> VERIFIED_TRUE (93%)
        5b. Raw Text Strongly Corroborated by Accredited Wire -> VERIFIED_TRUE (92%)
        6. General Tribrid Weighted Synthesis (Calibrated)
        """
        dom_cat = domain_info.get("category")
        dom_score = domain_info.get("trust_score", 0.5)

        debunk_summary = fact_consensus.get("debunk_summary", {})
        has_debunk = debunk_summary.get("has_debunk", False)
        debunk_verdict = debunk_summary.get("debunk_verdict")

        wire_consensus = fact_consensus.get("wire_consensus", {})
        wire_cnt = wire_consensus.get("wire_count", 0)

        ml_real_prob = stylometry.get("real_probability", 0.5)
        heuristics = stylometry.get("heuristics", {})
        suspicion = heuristics.get("heuristic_suspicion_score", 0.0)

        # Rule 1: Satire
        if dom_cat == "SATIRE" or debunk_verdict == "SATIRE":
            return (
                12.0,
                "SATIRE_PARODY",
                "HIGH",
                f"Source or content is confirmed satire/parody ({domain_info.get('details')}). Not intended as factual news."
            )

        # Rule 2: Flagged Disinformation Registry
        if dom_cat == "DISINFORMATION":
            return (
                6.0,
                "CONFIRMED_FAKE_OR_HOAX",
                "HIGH",
                f"Published on a confirmed disinformation/hoax domain ({domain_info.get('domain')}). Blacklisted by media watchdogs."
            )

        # Rule 3: Direct Fact-Check Debunk
        if has_debunk and debunk_verdict == "FALSE":
            top_fc = fact_consensus.get("fact_checks", [{}])[0]
            publisher = top_fc.get("publisher", "Independent Fact Checker")
            rating = top_fc.get("rating", "False")
            return (
                8.0,
                "CONFIRMED_FAKE_OR_HOAX",
                "HIGH",
                f"Directly debunked by {publisher} with rating '{rating}'."
            )

        # Rule 4: Direct Fact-Check Verified True
        if has_debunk and debunk_verdict == "VERIFIED_TRUE":
            return (
                94.0,
                "VERIFIED_TRUE",
                "HIGH",
                "Claims have been independently verified and rated accurate by accredited fact-checkers."
            )

        # Rule 5: Reputable Wire Outlet + Wire Consistency
        if dom_cat in ["REPUTABLE", "FACT_CHECKER"] and wire_cnt >= 1:
            return (
                93.0,
                "VERIFIED_TRUE",
                "HIGH",
                f"Published by accredited organization ({domain_info.get('domain')}) and corroborated across major news wires."
            )

        if dom_cat in ["REPUTABLE", "FACT_CHECKER"]:
            base_score = 82.0 + (ml_real_prob * 10.0) - (suspicion * 10.0)
            return (
                min(max(base_score, 75.0), 95.0),
                "LIKELY_REAL",
                "MEDIUM",
                f"Published by verified news outlet ({domain_info.get('domain')}) with standard journalistic formatting."
            )

        # Rule 5b: Raw Text Strongly Corroborated by Accredited News Wires
        reputable_sources = wire_consensus.get("reputable_sources", [])
        max_overlap = wire_consensus.get("max_overlap", 0.0)

        # Requires high semantic similarity (>= 0.45) to prevent generic word matches
        if len(reputable_sources) > 0 and max_overlap >= 0.45:
            source_str = ", ".join(reputable_sources[:3])
            return (
                92.0,
                "VERIFIED_TRUE",
                "HIGH",
                f"Story independently verified and corroborated by accredited news reporting ({source_str})."
            )

        # Rule 6: General Tribrid Weighted Synthesis
        consensus_score = fact_consensus.get("consensus_score", 0.5)

        if has_url:
            composite = (dom_score * 0.25) + (ml_real_prob * 0.40) + (consensus_score * 0.35)
        else:
            # For raw text without a domain:
            # If no wire consensus is found, do not falsely brand calm text as fake.
            if wire_cnt == 0:
                composite = (ml_real_prob * 0.65) + (consensus_score * 0.35)
            else:
                composite = (ml_real_prob * 0.45) + (consensus_score * 0.55)

        # Apply heuristic penalty if sensationalist clickbait triggers detected
        composite -= (suspicion * 0.35)
        composite = min(max(composite, 0.0), 1.0)
        final_score = composite * 100.0

        if final_score >= 78.0:
            verdict = "VERIFIED_TRUE"
            confidence = "HIGH" if wire_cnt > 0 else "MEDIUM"
            reason = "High journalistic probability with corroborated reporting and balanced linguistic tone."
        elif final_score >= 58.0:
            verdict = "LIKELY_REAL"
            confidence = "MEDIUM"
            reason = "Writing style and available signals lean credible, though independent confirmation is moderate."
        elif final_score >= 42.0:
            verdict = "UNVERIFIED_UNCERTAIN"
            confidence = "LOW" if wire_cnt == 0 else "MEDIUM"
            reason = "Insufficient external evidence or unindexed claim. Neutral linguistic tone, but verify through primary sources."
        elif final_score >= 22.0:
            verdict = "LIKELY_FAKE"
            confidence = "MEDIUM"
            reason = "Sensationalist linguistic patterns, unverified claims, or lack of credible media coverage."
        else:
            verdict = "CONFIRMED_FAKE_OR_HOAX"
            confidence = "HIGH"
            reason = "Strong hallmarks of manufactured disinfo: sensationalist tone, uncorroborated assertions, and dubious origins."

        return (final_score, verdict, confidence, reason)
