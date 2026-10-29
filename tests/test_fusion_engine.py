import pytest
from src.inference.fusion_engine import TribridEngine


@pytest.mark.asyncio
async def test_fusion_engine_satire_outlet():
    engine = TribridEngine()
    result = await engine.verify(url="https://theonion.com/article-slug-test")
    assert result.verdict == "SATIRE_PARODY"
    assert result.confidence_level == "HIGH"
    assert result.credibility_score <= 20.0


@pytest.mark.asyncio
async def test_fusion_engine_disinfo_outlet():
    engine = TribridEngine()
    result = await engine.verify(url="https://infowars.com/conspiracy-claim-test")
    assert result.verdict == "CONFIRMED_FAKE_OR_HOAX"
    assert result.confidence_level == "HIGH"
    assert result.credibility_score <= 10.0


@pytest.mark.asyncio
async def test_fusion_engine_reputable_outlet():
    engine = TribridEngine()
    result = await engine.verify(url="https://reuters.com/markets/rate-cut-decision-test")
    # Reputable domain with unparsed body returns LIKELY_REAL with paywall note
    assert result.verdict in ["VERIFIED_TRUE", "LIKELY_REAL"]
    assert result.credibility_score >= 75.0


@pytest.mark.asyncio
async def test_fusion_engine_raw_hoax_text():
    engine = TribridEngine()
    title = "SHOCKING BOMBSHELL: Proof that Aliens Run Secret Deep State"
    text = "YOU WON'T BELIEVE THIS! Secret aliens took over the media! Wake up!"
    result = await engine.verify(title=title, text=text)
    assert result.verdict in ["CONFIRMED_FAKE_OR_HOAX", "LIKELY_FAKE"]
    assert result.credibility_score <= 35.0

