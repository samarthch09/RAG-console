"""
benchmark.py — measure real latency of the running Index API.

Usage:
    python3 benchmark.py --url http://localhost:8000 --question "What is this document about?" --n 30

Requires the server to be running locally with at least one PDF already ingested
(otherwise queries will return "no context found" instantly and skew results low).
"""
import argparse
import statistics
import time
import sys

import httpx


def bench_query(base_url: str, question: str, top_k: int, n: int) -> list[float]:
    times = []
    with httpx.Client(timeout=60.0) as client:
        for i in range(n):
            start = time.perf_counter()
            resp = client.post(f"{base_url}/api/query", json={"question": question, "top_k": top_k})
            elapsed = time.perf_counter() - start
            if resp.status_code != 200:
                print(f"  [{i+1}/{n}] ERROR {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
                continue
            times.append(elapsed)
            print(f"  [{i+1}/{n}] {elapsed*1000:.0f}ms")
    return times


def bench_ingest(base_url: str, pdf_path: str) -> float:
    with httpx.Client(timeout=180.0) as client:
        with open(pdf_path, "rb") as f:
            start = time.perf_counter()
            resp = client.post(f"{base_url}/api/ingest", files={"file": (pdf_path.split("/")[-1], f, "application/pdf")})
            elapsed = time.perf_counter() - start
    if resp.status_code != 200:
        print(f"Ingest failed: {resp.status_code} {resp.text[:300]}", file=sys.stderr)
        return -1
    data = resp.json().get("data", {})
    print(f"Ingested {data.get('ingested', '?')} chunks in {elapsed:.2f}s")
    return elapsed


def summarize(label: str, times: list[float]):
    if not times:
        print(f"{label}: no successful samples")
        return
    times_sorted = sorted(times)
    n = len(times_sorted)
    p50 = times_sorted[int(n * 0.50) - 1] if n >= 2 else times_sorted[0]
    p95 = times_sorted[min(int(n * 0.95), n - 1)]
    p99 = times_sorted[min(int(n * 0.99), n - 1)]
    print(f"\n=== {label} (n={n}) ===")
    print(f"  mean: {statistics.mean(times)*1000:.0f}ms")
    print(f"  p50:  {p50*1000:.0f}ms")
    print(f"  p95:  {p95*1000:.0f}ms")
    print(f"  p99:  {p99*1000:.0f}ms")
    print(f"  min:  {min(times)*1000:.0f}ms")
    print(f"  max:  {max(times)*1000:.0f}ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--question", default="Summarize the key points of this document.")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--n", type=int, default=30, help="number of query requests to sample")
    ap.add_argument("--pdf", default=None, help="optional path to a PDF to also benchmark ingestion")
    args = ap.parse_args()

    if args.pdf:
        print(f"Benchmarking ingestion with {args.pdf} ...")
        bench_ingest(args.url, args.pdf)
        print()

    print(f"Benchmarking {args.n} queries against {args.url} ...")
    times = bench_query(args.url, args.question, args.top_k, args.n)
    summarize("Query latency (/api/query, end-to-end incl. embed+search+LLM)", times)