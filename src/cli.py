"""
CLI Interface for Real-Time Tribrid Fake News Detection Engine
Usage:
    python src/cli.py --url "https://www.reuters.com/..."
    python src/cli.py --text "BREAKING: Scientists discover secret cure..." --title "Secret Cure Found"
    python src/cli.py --interactive
"""

import sys
import asyncio
import argparse
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.inference.fusion_engine import TribridEngine, VerificationResult


def format_colored_verdict(verdict: str) -> str:
    # ANSI escape sequences
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    if verdict == "VERIFIED_TRUE":
        return f"{GREEN}{BOLD}[VERIFIED TRUE]{RESET}"
    elif verdict == "LIKELY_REAL":
        return f"{CYAN}{BOLD}[LIKELY REAL]{RESET}"
    elif verdict == "UNVERIFIED_UNCERTAIN":
        return f"{YELLOW}{BOLD}[UNVERIFIED / UNCERTAIN]{RESET}"
    elif verdict == "LIKELY_FAKE":
        return f"{YELLOW}{BOLD}[LIKELY FAKE / SENSATIONAL]{RESET}"
    elif verdict == "CONFIRMED_FAKE_OR_HOAX":
        return f"{RED}{BOLD}[CONFIRMED FAKE / HOAX]{RESET}"
    elif verdict == "SATIRE_PARODY":
        return f"{MAGENTA}{BOLD}[SATIRE / PARODY]{RESET}"
    return verdict


def print_report(res: VerificationResult):
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    print("\n" + "=" * 70)
    print(f"{CYAN}{BOLD} TRIBRID NEWS CREDIBILITY INTELLIGENCE REPORT{RESET}")
    print("=" * 70)
    
    if res.title:
        print(f"{BOLD}Title:{RESET} {res.title}")
    if res.url:
        print(f"{BOLD}URL:{RESET}   {res.url}")
    if res.author:
        print(f"{BOLD}Author:{RESET} {res.author}")
    if res.publish_date:
        print(f"{BOLD}Date:{RESET}   {res.publish_date}")
    print(f"{BOLD}Word Count:{RESET} {res.word_count} words")
    print("-" * 70)

    # Core Verdict
    badge = format_colored_verdict(res.verdict)
    bar_filled = int(res.credibility_score / 5)
    bar_empty = 20 - bar_filled
    gauge = "[" + "#" * bar_filled + "-" * bar_empty + "]"

    print(f"{BOLD}CREDIBILITY INDEX:{RESET} {res.credibility_score:5.1f}%  {gauge}")
    print(f"{BOLD}VERDICT:{RESET}           {badge}")
    print(f"{BOLD}CONFIDENCE:{RESET}        {res.confidence_level}")
    print(f"{BOLD}PRIMARY RATIONALE:{RESET} {res.primary_reason}")
    print("-" * 70)

    # Pillar 1
    p1 = res.pillar_1_domain
    print(f"{BOLD}[PILLAR 1: DOMAIN REPUTATION & PROVENANCE]{RESET}")
    print(f"  * Domain:     {p1.get('domain', 'N/A')}")
    print(f"  * Status:     {p1.get('status_label', 'N/A')}")
    print(f"  * Trust:      {p1.get('trust_score', 0.5) * 100:.1f}%")
    print(f"  * Note:       {p1.get('details', 'N/A')}")

    # Pillar 2
    p2 = res.pillar_2_stylometry
    heuristics = p2.get("heuristics", {})
    print(f"\n{BOLD}[PILLAR 2: STYLOMETRIC NLP & ML CLASSIFIER]{RESET}")
    print(f"  * ML Real Probability:      {p2.get('real_probability', 0.5) * 100:.1f}%")
    print(f"  * ML Fake Probability:      {p2.get('fake_probability', 0.5) * 100:.1f}%")
    print(f"  * Stylometric Style:        {p2.get('stylometric_label', 'N/A')}")
    print(f"  * Uppercase Word Ratio:     {heuristics.get('caps_ratio', 0.0) * 100:.2f}%")
    print(f"  * Exclamation Density:      {heuristics.get('exclamation_density', 0.0)} per 1k chars")
    sens = heuristics.get("sensational_hits", [])
    if sens:
        print(f"  * Sensationalism Triggers:  {', '.join(sens)}")
    else:
        print(f"  * Sensationalism Triggers:  None detected (Journalistic tone)")
    top_words = p2.get("top_features", [])
    if top_words:
        print(f"  * Key Predictive Features:  {', '.join(top_words[:6])}")

    # Pillar 3
    p3 = res.pillar_3_fact_consensus
    wires = p3.get("wire_consensus", {})
    fchecks = p3.get("fact_checks", [])
    print(f"\n{BOLD}[PILLAR 3: LIVE FACT CONSENSUS & WIRE SEARCH]{RESET}")
    print(f"  * Search Query:             \"{p3.get('query', 'N/A')}\"")
    print(f"  * Wire Mentions (Reuters/AP/BBC): {wires.get('wire_count', 0)} wire(s) corroborating")
    if fchecks:
        print(f"  * Fact-Check Records Found: {len(fchecks)}")
        for fc in fchecks[:2]:
            print(f"    - [{fc.get('publisher')}] Rating: '{fc.get('rating')}' -> {fc.get('claim')[:60]}...")
    else:
        print(f"  * Fact-Check Records:       No debunked hoaxes recorded for this specific query")

    print("-" * 70)
    print(f"Execution Latency: {res.execution_time_ms:.1f} ms")
    print("=" * 70 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="Real-Time Tribrid Fake News Detection Engine")
    parser.add_argument("--url", type=str, help="Full news article URL to analyze")
    parser.add_argument("--text", type=str, help="News article text or claim")
    parser.add_argument("--title", type=str, default="", help="Optional article headline")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prompt")

    args = parser.parse_args()

    engine = TribridEngine()

    if args.interactive:
        print("\n=== Real-Time Fake News Verification CLI ===")
        print("Type 'exit' or 'quit' at any prompt to stop.\n")
        while True:
            mode = input("Verify by (1) URL or (2) Raw Text? [1/2]: ").strip()
            if mode in ["exit", "quit"]:
                break
            if mode == "1":
                url = input("Enter Article URL: ").strip()
                if url in ["exit", "quit"]:
                    break
                print("\n[Analyzing via Tribrid Pipeline...]")
                res = await engine.verify(url=url)
                print_report(res)
            elif mode == "2":
                title = input("Enter Article Title (or press Enter): ").strip()
                print("Enter Article Body Text (Finish with Enter):")
                text = input("> ").strip()
                if not text:
                    print("Error: Article text cannot be empty.\n")
                    continue
                print("\n[Analyzing via Tribrid Pipeline...]")
                res = await engine.verify(text=text, title=title)
                print_report(res)
            else:
                print("Invalid choice. Please select 1 or 2.")
    elif args.url or args.text:
        res = await engine.verify(url=args.url, text=args.text, title=args.title)
        print_report(res)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
