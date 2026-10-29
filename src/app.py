"""
FastAPI Server for Real-Time Tribrid Fake News Detection Engine
Hosts REST API endpoints and serves the modern single-page dashboard.
Hardened with CORS middleware, payload size validation, and refined metadata.
"""

from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference.fusion_engine import TribridEngine, VerificationResult
from src.config import MAX_TEXT_LENGTH, MAX_TITLE_LENGTH, HOST, PORT

app = FastAPI(
    title="Tribrid Fake News Intelligence Engine",
    description="Real-time multi-layered news credibility verification engine combining Domain Reputation, Stylometric NLP, and Live Fact Consensus.",
    version="2.1.0"
)

# Enable CORS for external client applications and dashboard embeds
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

engine = TribridEngine()


class VerifyRequest(BaseModel):
    url: Optional[str] = Field(None, description="Direct URL of the article to scrape and verify")
    text: Optional[str] = Field(None, max_length=MAX_TEXT_LENGTH, description=f"Raw text of the news article or claim (max {MAX_TEXT_LENGTH:,} chars)")
    title: Optional[str] = Field("", max_length=MAX_TITLE_LENGTH, description=f"Optional article headline (max {MAX_TITLE_LENGTH:,} chars)")


@app.get("/")
async def serve_index():
    """Serves the main single-page dashboard."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend dashboard not found.")
    return FileResponse(index_file)


@app.get("/api/health")
async def health_check():
    """System health check and module status."""
    return {
        "status": "healthy",
        "engine": "Tribrid Fusion v2.1 (Hardened)",
        "pillars": {
            "pillar_1": "Domain Reputation & Provenance (Curated Media Bias / IFCN Desks)",
            "pillar_2": "Stylometric NLP (Complement Naive Bayes + Heuristics)",
            "pillar_3": "Live Fact Consensus (ClaimReview & Wire Search)"
        }
    }


@app.post("/api/verify", response_model=VerificationResult)
async def verify_news(req: VerifyRequest):
    """
    Verifies a news item either by URL or raw text.
    Executes the 3-pillar Tribrid waterfall inference pipeline.
    """
    if not req.url and not (req.text and req.text.strip()):
        raise HTTPException(status_code=400, detail="Must provide either an article URL or text content.")

    try:
        result = await engine.verify(url=req.url, text=req.text, title=req.title)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host=HOST, port=PORT, reload=True)
