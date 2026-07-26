"""
Lightweight tests for the FastAPI surface that don't require live OpenAI/Qdrant
credentials. External calls (embeddings, vector store) are monkeypatched.
"""
import io
import sys
import types
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, "..")


@pytest.fixture
def client(monkeypatch):
    # Stub out network-dependent modules before importing main.
    fake_store = MagicMock()
    fake_store.health.return_value = True
    fake_store.stats.return_value = {"points_count": 12, "vectors_count": 12, "status": "green", "dim": 3072}
    fake_store.list_sources.return_value = ["handbook.pdf"]

    with patch("vector_db.QdrantStorage", return_value=fake_store):
        from fastapi.testclient import TestClient
        import main
        yield TestClient(main.app)


def test_health_ok(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["data"]["status"] == "ok"


def test_stats_shape(client):
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "points_count" in data
    assert "sources" in data


def test_ingest_rejects_non_pdf(client):
    resp = client.post(
        "/api/ingest",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_file_type"


def test_query_validates_empty_question(client):
    resp = client.post("/api/query", json={"question": "", "top_k": 5})
    assert resp.status_code == 422


def test_query_validates_top_k_bounds(client):
    resp = client.post("/api/query", json={"question": "hi", "top_k": 999})
    assert resp.status_code == 422
