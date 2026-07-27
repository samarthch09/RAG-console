# Index — RAG Console

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector--search-DC244C?style=flat-square)](https://qdrant.tech/)

> Upload a PDF. Ask it questions. Get answers grounded in cited passages, not guesses.

**Index** is a full-stack retrieval-augmented generation (RAG) service: a durable, event-driven ingestion pipeline, a REST API with a consistent response contract, and a dependency-free web console for uploading documents and querying them in real time.

This started from a minimal reference implementation and was rebuilt end-to-end: a new REST API layer, a fully custom frontend (no framework, no build step), input validation and structured error handling, health/observability endpoints, containerization, and a test suite.

---

## Table of Contents

- [Why this exists](#why-this-exists)
- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Design Decisions](#design-decisions)
- [Production Considerations](#production-considerations)
- [Roadmap](#roadmap)
- [License](#license)

---

## Why this exists

Most RAG demos stop at "upload a file, print an answer in a notebook." This project treats it as a small production system instead:

- Ingestion is a **durable background job** (via Inngest), not a blocking HTTP request — so a slow embedding call or a rate limit doesn't take down the API.
- The **REST layer has one response contract** (`{"data": ...}` / `{"error": {code, message}}`) so the frontend never has to special-case endpoints.
- The **frontend is a single dependency-free HTML file** that talks only to documented REST endpoints — anyone could swap it for React/Vue without touching the backend.
- Every write path validates input (file type, file size, question length, `top_k` bounds) and fails with a specific, actionable error.

---

## Architecture

```
┌─────────────┐      REST (/api/*)      ┌──────────────────┐
│  Web Console │ ───────────────────────▶│     FastAPI       │
│ (static SPA) │◀─────────────────────── │   (main.py)        │
└─────────────┘        JSON              └─────────┬─────────┘
                                                     │ sends event
                                                     ▼
                                          ┌──────────────────┐
                                          │  Inngest Worker    │
                                          │ (durable steps,     │
                                          │  retries, rate       │
                                          │  limits, throttling) │
                                          └─────────┬────────┘
                                    ┌────────────────┼────────────────┐
                                    ▼                ▼                ▼
                          ┌─────────────┐  ┌──────────────┐  ┌───────────────┐
                          │  PDF Parser  │  │  OpenAI       │  │   Qdrant        │
                          │ + Chunker    │  │  Embeddings   │  │  Vector Store   │
                          │ (LlamaIndex) │  │  + gpt-4o-mini│  │  (cosine sim)   │
                          └─────────────┘  └──────────────┘  └───────────────┘
```

**Request flow:**
1. Browser uploads a PDF (or asks a question) via `/api/ingest` or `/api/query`.
2. FastAPI sends an event to Inngest and **polls the run status** server-side — the browser only ever talks to FastAPI.
3. Inngest executes the pipeline as discrete, retryable steps: parse & chunk → embed → upsert (ingestion), or embed query → vector search → LLM synthesis (query).
4. The result is returned through the same REST envelope the frontend already understands.

---

## Features

**Backend**
- Async FastAPI service with a typed, validated REST API (`pydantic` request models)
- Durable ingestion/query pipeline via Inngest: automatic retries, per-source rate limiting, throttling
- Structured error handling with machine-readable error codes (`invalid_file_type`, `file_too_large`, `run_failed`, `timeout`, …)
- `/api/health` and `/api/stats` endpoints for uptime checks and index observability
- File-size limits and content-type validation on upload
- CORS configured for standalone frontend deployment

**Frontend**
- Single static HTML file, zero build step, zero framework dependency
- Drag-and-drop PDF upload with a **live animated pipeline visualizer** (upload → chunk → embed → index)
- Chat-style query interface with per-answer **cited source chips**
- Real-time backend health indicator and index dashboard (chunk count, document count, vector dimension)
- Toast notifications, keyboard shortcuts (Enter to send), responsive layout down to mobile

**Engineering**
- `pytest` suite covering API contracts and vector-store helpers, with external services mocked
- Dockerfile + `docker-compose.yml` wiring Qdrant, the Inngest dev server, and the API together
- `.env.example` documenting every configuration key

---

## Quick Start

### Option A — Docker (recommended)

```bash
cp .env.example .env        # fill in OPENAI_API_KEY
docker compose up --build
```

Open **http://localhost:8000** — the console is served directly from the API.

### Option B — Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in OPENAI_API_KEY

# Terminal 1 — vector store
docker run -p 6333:6333 qdrant/qdrant

# Terminal 2 — Inngest dev server
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest

# Terminal 3 — API + frontend
uvicorn main:app --reload
```

Open **http://localhost:8000**.

---

## Configuration

All configuration lives in environment variables (see `.env.example`):

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | Embeddings + chat completion | — (required) |
| `QDRANT_URL` | Vector store endpoint | `http://localhost:6333` |
| `QDRANT_API_KEY` | Vector store auth (optional) | — |
| `INNGEST_API_BASE` | Inngest dev server REST API | `http://127.0.0.1:8288/v1` |
| `MAX_UPLOAD_MB` | Upload size limit | `25` |
| `EMBED_DIM` | Embedding vector dimension | `3072` |
| `CORS_ORIGINS` | Allowed origins, comma-separated | `*` |

---

## API Reference

All responses use the envelope `{"data": ...}` on success or `{"error": {"code": ..., "message": ...}}` on failure.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Backend + Qdrant connectivity status |
| `GET` | `/api/stats` | Chunk count, indexed document list, vector dimension |
| `POST` | `/api/ingest` | Multipart PDF upload → chunk, embed, index (blocks until complete) |
| `POST` | `/api/query` | `{"question": str, "top_k": int}` → grounded answer + cited sources |

```bash
curl -X POST http://localhost:8000/api/ingest -F "file=@handbook.pdf"

curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the escalation process for a P1 incident?", "top_k": 5}'
```

---

## Project Structure

```
.
├── main.py            # FastAPI app: REST endpoints + Inngest pipeline definitions
├── data_loader.py      # PDF parsing, chunking, embedding
├── vector_db.py         # Qdrant client wrapper (upsert, search, stats, health)
├── custom_types.py       # Pydantic models shared across the pipeline
├── frontend/index.html    # Static single-page console (no build step)
├── tests/                  # pytest suite, external services mocked
├── docker-compose.yml       # Qdrant + Inngest + API, wired together
├── Dockerfile
└── .env.example
```

---

## Testing

```bash
pip install pytest
cd tests && pytest -v
```

Tests mock `QdrantClient` and the embedding calls so the suite runs without any live credentials or network access — useful in CI.

---

## Design Decisions

- **Why poll instead of push?** The frontend never talks to Inngest directly, which keeps the browser's attack surface to a single trusted origin (the FastAPI app) and lets the backend own retry/timeout semantics in one place.
- **Why a static frontend instead of a framework?** For a project this size, a build pipeline adds complexity without adding capability. The console is one file, has zero dependencies, and can be swapped for a React app later without changing a single backend contract.
- **Why a consistent error envelope?** So the frontend's error handling is one function (`api()`) instead of a special case per endpoint — the same pattern scales to a much larger API surface.

---

## Production Considerations

- **Scaling** — run Qdrant as a cluster; run the Inngest worker separately from the API for independent scaling of ingestion vs. query load.
- **Security** — validate and scan uploads, enforce size limits (already implemented), and move secrets to a manager (AWS Secrets Manager, Vault) instead of `.env` in production.
- **Observability** — `/api/health` and `/api/stats` are designed to back an uptime check and a metrics dashboard (Prometheus scrape target is a natural next step).
- **Cost** — batch embedding calls where possible and monitor OpenAI usage; the throttling/rate-limit config on the Inngest function already caps ingestion frequency per source.

## Roadmap

- [ ] Streaming answers (token-by-token) over SSE
- [ ] Multi-file batch ingestion with a progress queue
- [ ] Per-document deletion from the index
- [ ] Prometheus metrics endpoint

---


