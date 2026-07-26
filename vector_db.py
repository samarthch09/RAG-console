import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:
    def __init__(self, url: str | None = None, collection: str = "docs", dim: int = 3072):
        url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY") or None
        self.client = QdrantClient(url=url, api_key=api_key, timeout=30)
        self.collection = collection
        self.dim = dim
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def stats(self) -> dict:
        """Return collection-level stats used to power the dashboard."""
        info = self.client.get_collection(self.collection)
        return {
            "points_count": info.points_count or 0,
            "vectors_count": info.vectors_count or 0,
            "status": str(info.status),
            "dim": self.dim,
        }

    def list_sources(self) -> list[str]:
        """Best-effort distinct list of ingested source documents."""
        sources = set()
        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=next_offset,
            )
            for p in points:
                src = (p.payload or {}).get("source")
                if src:
                    sources.add(src)
            if next_offset is None:
                break
        return sorted(sources)

    def health(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k: int = 5):
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            with_payload=True,
            limit=top_k
        )
        contexts = []
        sources = set()

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}