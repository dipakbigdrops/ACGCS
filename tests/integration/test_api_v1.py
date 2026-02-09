import io
import os
import pytest


@pytest.mark.asyncio
async def test_root(client):
    r = await client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "message" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


@pytest.mark.asyncio
async def test_v1_health(client):
    r = await client.get("/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_v1_root(client):
    r = await client.get("/v1/")
    assert r.status_code == 200
    data = r.json()
    assert "version" in data


@pytest.mark.asyncio
async def test_upload_guidelines_no_file_uses_default(client):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    default_path = os.path.join(base, "default_guidelines.pdf")
    if not os.path.exists(default_path):
        pytest.skip("default_guidelines.pdf not found")
    r = await client.post("/v1/upload-guidelines")
    assert r.status_code == 200
    data = r.json()
    assert "guidelines_id" in data
    assert "rules_count" in data


@pytest.mark.asyncio
async def test_upload_and_analyze(client):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    default_path = os.path.join(base, "default_guidelines.pdf")
    if not os.path.exists(default_path):
        pytest.skip("default_guidelines.pdf not found")
    r = await client.post("/v1/upload-guidelines")
    if r.status_code != 200:
        pytest.skip("upload failed")
    data = r.json()
    guidelines_id = data["guidelines_id"]
    html_content = b"<html><body><p>Hello world</p></body></html>"
    r2 = await client.post(
        "/v1/analyze",
        data={"guidelines_id": guidelines_id},
        files={"creative_file": ("t.html", io.BytesIO(html_content), "text/html")},
    )
    assert r2.status_code == 200
    resp = r2.json()
    assert "message" in resp


@pytest.mark.asyncio
async def test_analyze_unsupported_type(client):
    r = await client.post(
        "/v1/analyze",
        data={"guidelines_id": ""},
        files={"creative_file": ("x.txt", io.BytesIO(b"text"), "text/plain")},
    )
    assert r.status_code == 400
    assert "Unsupported" in str(r.json().get("detail", ""))


@pytest.mark.asyncio
async def test_guidelines_list(client):
    r = await client.get("/v1/guidelines?page=1&page_size=5")
    assert r.status_code == 200
    data = r.json()
    assert "guidelines_ids" in data
    assert "total" in data
    assert "page" in data
    assert data["page"] == 1


@pytest.mark.asyncio
async def test_metrics(client):
    r = await client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "request_count" in data or "latency_seconds" in data or "request_errors" in data
