"""
RAG Production App — FastAPI backend.

Exposes:
  * Background ingestion/query pipeline via Inngest (durable, retried, rate-limited)
  * A synchronous REST API used by the web frontend (/api/*), which triggers the
    same Inngest functions and waits for the run to complete
  * A static file mount that serves the SPA frontend

Design notes:
  * All /api endpoints return a consistent JSON envelope: {"data": ...} on success
    or {"error": {"code": ..., "message": ...}} on failure, so the frontend can
    handle every response the same way.
  * Long-running ingestion/query work stays on the Inngest worker; the REST layer
    only polls for the result, so the browser never talks to Inngest directly.
"""
import logging
import os
import time
import uuid
import datetime
from pathlib import Path

import httpx
import inngest
import inngest.fast_api
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

from custom_types import RAGChunkAndSrc, RAGSearchResult, RAGUpsertResult
from data_loader import embed_texts, load_and_chunk_pdf
from vector_db import QdrantStorage

load_dotenv()

logger = logging.getLogger("uvicorn")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
EMBED_DIM = int(os.getenv("EMBED_DIM", "3072"))

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logger,
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


# ---------------------------------------------------------------------------
# Background functions (Inngest)
# ---------------------------------------------------------------------------

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(limit=2, period=datetime.timedelta(minutes=1)),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
    ),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        if not chunks:
            raise ValueError(f"No extractable text found in '{source_id}'")
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [
            {"source": source_id, "text": chunks[i], "chunk_index": i, "ingested_at": time.time()}
            for i in range(len(chunks))
        ]
        QdrantStorage(dim=EMBED_DIM).upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage(dim=EMBED_DIM)
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    if not found.contexts:
        return {
            "answer": "I couldn't find anything relevant in the indexed documents yet. "
                      "Try ingesting a PDF first, or rephrase your question.",
            "sources": [],
            "num_contexts": 0,
        }

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above. If the context does not contain "
        "the answer, say so explicitly rather than guessing."
    )

    def _generate_answer() -> str:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1024,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content},
            ],
        )
        return res.choices[0].message.content.strip()

    answer = await ctx.step.run("llm-answer", _generate_answer)
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Production App",
    description="Retrieval-augmented generation service for querying indexed PDF documentation.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message


@app.exception_handler(ApiError)
async def api_error_handler(request, exc: ApiError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


def _ok(data):
    return {"data": data}


async def _run_and_wait(event_name: str, data: dict, timeout_s: float = 90.0, poll_interval_s: float = 0.4) -> dict:
    """Send an Inngest event and poll the run-status API until it finishes."""
    ids = await inngest_client.send(inngest.Event(name=event_name, data=data))
    event_id = ids[0]

    base = os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")
    start = time.time()
    last_status = None

    async with httpx.AsyncClient(timeout=10.0) as http:
        while True:
            resp = await http.get(f"{base}/events/{event_id}/runs")
            resp.raise_for_status()
            runs = resp.json().get("data", [])
            if runs:
                run = runs[0]
                status = run.get("status")
                last_status = status or last_status
                if status in ("Completed", "Succeeded", "Success", "Finished"):
                    return run.get("output") or {}
                if status in ("Failed", "Cancelled"):
                    raise ApiError(502, "run_failed", f"Background job {status.lower()}: {run.get('output')}")
            if time.time() - start > timeout_s:
                raise ApiError(504, "timeout", f"Timed out waiting for job (last status: {last_status})")
            time.sleep(poll_interval_s)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)


@app.get("/api/health")
async def health():
    qdrant_ok = QdrantStorage(dim=EMBED_DIM).health()
    return _ok({
        "status": "ok" if qdrant_ok else "degraded",
        "qdrant": "up" if qdrant_ok else "down",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "time": time.time(),
    })


@app.get("/api/stats")
async def stats():
    try:
        store = QdrantStorage(dim=EMBED_DIM)
        info = store.stats()
        info["sources"] = store.list_sources()
        return _ok(info)
    except Exception as exc:  # pragma: no cover - defensive
        raise ApiError(503, "qdrant_unavailable", str(exc))


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise ApiError(400, "invalid_file_type", "Only PDF files are supported.")

    dest = UPLOAD_DIR / file.filename
    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                out.close()
                dest.unlink(missing_ok=True)
                raise ApiError(413, "file_too_large", f"File exceeds {MAX_UPLOAD_MB}MB limit.")
            out.write(chunk)

    try:
        output = await _run_and_wait(
            "rag/ingest_pdf",
            {"pdf_path": str(dest.resolve()), "source_id": file.filename},
        )
    except ApiError:
        raise
    except Exception as exc:
        raise ApiError(500, "ingest_failed", str(exc))

    return _ok({"filename": file.filename, "size_bytes": size, **output})


@app.post("/api/query")
async def query(req: QueryRequest):
    try:
        output = await _run_and_wait("rag/query_pdf_ai", {"question": req.question, "top_k": req.top_k})
    except ApiError:
        raise
    except Exception as exc:
        raise ApiError(500, "query_failed", str(exc))
    return _ok(output)


# ---------------------------------------------------------------------------
# Frontend hosting
# ---------------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")

    @app.get("/")
    async def index():
        return FileResponse(FRONTEND_DIR / "index.html")