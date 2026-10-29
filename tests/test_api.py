import pytest
import httpx
from src.app import app


@pytest.mark.asyncio
async def test_api_health_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "pillars" in data


@pytest.mark.asyncio
async def test_api_verify_missing_parameters():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/verify", json={})
        assert response.status_code == 400
        assert "detail" in response.json()


@pytest.mark.asyncio
async def test_api_verify_valid_claim():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "title": "Central banks coordinate monetary policy across eurozone",
            "text": "European central banks held emergency discussions regarding inflation benchmarks."
        }
        response = await client.post("/api/verify", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "credibility_score" in data
        assert "verdict" in data
        assert "pillar_1_domain" in data
        assert "pillar_2_stylometry" in data
        assert "pillar_3_fact_consensus" in data
