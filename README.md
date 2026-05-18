# Real-Time Personalised Search & Recommendation API

![CI](https://github.com/AndrewZhao86/Click-Recommendation-System/actions/workflows/ci.yml/badge.svg)

A production-grade, end-to-end recommendation and personalised search engine backed by Kafka, Redis, Postgres/pgvector, and an optional LLM re-ranker. Demonstrates the full write path (clicks → Kafka → consumers → features) decoupled from the read path (search → candidate gen → feature lookup → rank → respond).

See [planning/plan.md](planning/plan.md) for the full design and phase breakdown.

---

## Architecture

```
Client ──POST /events/*──► FastAPI ──publish──► Kafka (6 partitions, KRaft)
Client ──GET /search,/recommendations──► FastAPI          │
                                             │             ▼
                                             │        Consumer group (×N)
                                             │             │
                                             ▼             ▼
                                           Redis ◄── enrichment ──► Postgres
                                             │                       (pgvector)
                                             └────────────────────────────────► LLM (Gemini / Groq / Ollama)
```

**Write path:** `POST /events/*` → Kafka → async consumer group → Redis feature updates + co-click matrix in Postgres. At-least-once delivery, idempotent consumers keyed on `event_id`, DLQ for poison pills.

**Read path:** `GET /search` → BM25 + pgvector ANN candidate gen → feature extraction (popularity, recency, personalisation, co-click, price-fit) → weighted linear ranker → optional LLM re-rank → top-10 with `score_breakdown`.

---

## Tech Stack

| Concern | Choice |
|---|---|
| API | FastAPI + Pydantic v2 + Uvicorn |
| Streaming | Apache Kafka (KRaft, no ZooKeeper) via `aiokafka` |
| Cache | Redis 7 — sorted sets, stampede-protection locks, cache-aside |
| Primary store | Postgres 16 + `pgvector` — BM25 `tsvector` + `ivfflat` ANN index |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| LLM | Gemini 2.5 Flash / Groq llama-3.1-8b / Ollama (swappable via `LLM_PROVIDER`) |
| Observability | OpenTelemetry (stdout + Jaeger), Prometheus `/metrics`, `structlog` JSON logs |
| Load testing | Locust |
| Tests | `pytest` + `pytest-asyncio` + `testcontainers` (real Kafka/Redis/Postgres) |
| Container | Multi-stage Dockerfile, non-root, model baked in; Trivy CVE scan in CI |

---

## Quickstart (host dev)

```bash
cp .env.example .env
make install        # uv sync --extra dev
make up             # Kafka (KRaft) + Kafka UI + Redis + Postgres (pgvector)
make migrate        # run Alembic migrations
make seed           # 10 k SKUs, 1 k users, embeddings → artifacts/seeded_users.txt
make api            # uvicorn on :8000
make consumer N=3   # 3 enrichment workers (each owns 2 of 6 partitions)
```

Smoke test:

```bash
curl http://localhost:8000/health
curl "http://localhost:8000/search?q=wireless+headphones&user_id=u_brand_loyalist_0001"
curl "http://localhost:8000/recommendations?user_id=u_brand_loyalist_0001"
```

Kafka UI: `http://localhost:8080`

---

## Quickstart (full stack in Docker)

Requires `GEMINI_API_KEY` (or leave blank to run LLM-less) in `.env`.

```bash
make docker-build   # builds click-rec:local (multi-stage, non-root, model baked in)
make docker-up      # infra + containerised API on :8000 (profile=app)
curl -f http://localhost:8000/health
make docker-logs
make docker-down
```

The `app` service is gated behind the `app` compose profile, so `make up` + `make api` (host-dev workflow) is unaffected.

Trivy CVE scanning runs in CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) on every push.

---

## Key Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/search` | `?q=&user_id=&use_llm=false` — personalised top-10 with `score_breakdown` |
| `GET` | `/recommendations` | `?user_id=` — for-you recommendations (no query) |
| `GET` | `/explain` | `?user_id=&item_id=` — NL rationale for a recommendation |
| `POST` | `/events/click` | Ingest a click event (202 + `event_id`) |
| `POST` | `/events/impression` | Ingest an impression event |
| `POST` | `/events/search` | Ingest a search event |
| `POST` | `/events/batch` | Batch ingest (≤ 500 events) |
| `GET` | `/metrics` | Prometheus metrics |

---

## Hybrid Ranker

Candidate generation (≈ 200 candidates per query):
- **BM25** — Postgres `tsvector` full-text search on item titles
- **Vector ANN** — `pgvector <->` cosine search on query embedding (MiniLM-L6-v2)

Per-candidate features extracted and linearly combined (weights in [`ranker.yaml`](ranker.yaml)):

| Feature | Source |
|---|---|
| `popularity_score` | Redis sorted set, decayed per-category |
| `recency_score` | Exponential decay on `item.created_at` |
| `personal_score` | Cosine similarity between item embedding and user's avg recent-click embedding |
| `co_click_score` | Postgres `co_click` join on user's last clicked item |
| `price_fit` | Distance from user's average price bucket (Redis profile hash) |

Optional LLM re-rank (Gemini / Groq / Ollama) re-orders top-20 with per-item rationale. Hard 2 s timeout falls back to hybrid order transparently.

---

## LLM Layer

Three use cases, all gated by `LLM_PROVIDER` env var and a 2 s timeout:

- **Query understanding** — parses vague queries into structured attributes (`category`, `attrs`, `price_bias`), cached in Redis 5 min by query hash.
- **Top-K re-ranker** — re-orders the hybrid top-20 with one-line rationale per item. Structured JSON output.
- **Explain** — `GET /explain` returns a 1–2 sentence natural-language justification referencing the user's click history.

Offline evaluation harness (`make eval-llm`) runs 100-tuple golden-set queries through the ranker; a separate judge LLM scores output quality in CI on every prompt change.

---

## Resilience & Distributed Systems Properties

| Scenario | Behaviour |
|---|---|
| Redis unavailable | Search falls back to Postgres-only; `cache_unavailable` logged; no 5xx |
| LLM timeout / quota exceeded | Falls back to hybrid ranker order; `llm_fallback_total` counter incremented |
| Kafka unavailable | API returns 503 with `Retry-After` header |
| Consumer killed mid-replay | Rebalance completes < 30 s; no message loss (manual offset commit after side-effects) |
| Poison-pill event | 3 retries + exponential back-off → published to `user.clicks.dlq`; consumer keeps running |
| Duplicate `event_id` | Redis `SET NX` on producer and consumer side; exactly one write |

DLQ replay: `make replay-dlq` reads `user.clicks.dlq`, fixes payload, re-publishes to main topic.

---

## Observability

```bash
curl localhost:8000/metrics | grep -E '(http_request_duration|cache_hit|kafka_consumer_lag|llm_)'
```

Key series: `http_request_duration_seconds` (p95), `cache_hit_total`, `cache_miss_total`, `cache_lock_wait_seconds`, `kafka_consumer_lag`, `llm_tokens_total`, `llm_fallback_total`.

Every log line carries `event_id`, `user_id`, and `trace_id` — a single `event_id` can be grepped across API and consumer logs to reconstruct its full path.

OTEL spans: `GET /search` → SQLAlchemy (BM25) → SQLAlchemy (ANN) → Redis → optional LLM call.

---

## Load Test Results

From `make loadtest` (500 RPS sustained, 5 min, LLM off):

- Zero 5xx
- p95 `/search` < 150 ms
- Consumer lag never exceeded 1 000
- Cache hit ratio > 90 % throughout

Scalability sweep (`make loadtest-sweep`) confirms near-linear throughput increase N=1 → N=3 → N=6 consumers (capped at 6 by Kafka partition count).

Full artefacts: [`artifacts/`](artifacts/) and [`docs/phase8_artifacts.md`](docs/phase8_artifacts.md).

---

## Scripts & Tools

| Command | Description |
|---|---|
| `make seed` | Seed 10 k SKUs + 1 k users + MiniLM embeddings |
| `make replay EVENTS=50000` | Simulate realistic click sessions into Kafka |
| `make replay-dlq` | Replay messages from DLQ back to main topic |
| `make loadtest` | Locust 500 RPS, 5 min (80 % search / 15 % click / 5 % impression) |
| `make loadtest-sweep` | Throughput vs consumer count (1 / 3 / 6 workers) |
| `make eval-llm` | NDCG@10 golden-set evaluation with LLM-as-judge |
| `make eval-offline` | Offline NDCG@10 + MRR@10 hybrid vs BM25 baseline |
| `uv run python scripts/proxy.py` | Round-robin reverse proxy over N single-worker uvicorns (Windows load-test workaround) |

---

## Project Structure

```
src/click_rec/
  api/          FastAPI app, routers (events, search, recommendations, explain, items)
  cache/        Redis client, cache-aside helper, stampede lock, popularity refresher
  db/           Async SQLAlchemy engine, Alembic migrations
  eval/         LLM-as-judge harness, offline NDCG eval
  kafka/        Async producer, consumer group, topic admin, enrichment logic
  llm/          Provider-agnostic LLMClient (Gemini / Groq / Ollama), re-ranker, explain
  models/       SQLAlchemy + Pydantic models (Item, User, ClickEvent, SearchQuery, CoClick)
  ranker/       Candidate gen (BM25 + ANN), feature extraction, weighted scorer, pipeline
  telemetry/    OTEL tracing, Prometheus metrics, structlog JSON logging
scripts/
  seed_catalog.py     Generate and embed 10 k SKUs + 1 k users
  replay_clicks.py    Simulate user sessions via Kafka
  replay_dlq.py       Re-publish DLQ messages to main topic
  throughput_sweep.py Measure max RPS at N=1/3/6 consumers
docs/
  phase8_artifacts.md  Load-test artefact guide and pass criteria
artifacts/             Locust CSVs, throughput_vs_consumers.csv, consumer logs
```
