"""Unit tests for QdrantStorage helpers, isolated from a real Qdrant instance."""
import sys
sys.path.insert(0, "..")
from unittest.mock import MagicMock, patch


def test_stats_reads_collection_info():
    with patch("vector_db.QdrantClient") as MockClient:
        instance = MockClient.return_value
        instance.collection_exists.return_value = True
        instance.get_collection.return_value = MagicMock(points_count=5, vectors_count=5, status="green")

        from vector_db import QdrantStorage
        store = QdrantStorage(url="http://fake:6333", dim=8)
        result = store.stats()
        assert result["points_count"] == 5
        assert result["dim"] == 8


def test_list_sources_deduplicates():
    with patch("vector_db.QdrantClient") as MockClient:
        instance = MockClient.return_value
        instance.collection_exists.return_value = True
        point_a = MagicMock(payload={"source": "a.pdf"})
        point_b = MagicMock(payload={"source": "a.pdf"})
        instance.scroll.return_value = ([point_a, point_b], None)

        from vector_db import QdrantStorage
        store = QdrantStorage(url="http://fake:6333")
        assert store.list_sources() == ["a.pdf"]
