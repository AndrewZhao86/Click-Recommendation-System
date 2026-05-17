# SearchPulse

![CI](https://github.com/AndrewZhao86/Click-Recommendation-System/actions/workflows/ci.yml/badge.svg)

Real-time, personalised product-search & recommendation engine.
Kafka-backed click ingestion, Redis hot-cache, hybrid ranker (BM25 + pgvector + co-click), optional LLM re-rank.

See [planning/plan.md](planning/plan.md) for the full design and phase breakdown.

## Quickstart (host dev)

```bash
cp .env.example .env
make install        # uv sync --extra dev
make up             # kafka (KRaft) + kafka-ui + redis + postgres (pgvector)
make test           # smoke: GET /health
make api            # uvicorn on :8000
```

## Quickstart (full stack in Docker)

Requires `GEMINI_API_KEY` set in a `.env` file at the repo root (compose auto-loads it; the `app` service fails loudly if it's unset).

```bash
make docker-build   # builds click-rec:local (multi-stage, non-root, model baked in)
make docker-up      # infra + containerised API on :8000 (profile=app)
curl -f http://localhost:8000/health
make docker-logs    # tail app logs
make docker-down
```

The `app` service is gated behind the `app` compose profile, so the existing host-dev workflow (`make up` + `make api`) is unaffected.

Trivy CVE scanning runs in CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) on every push — Linux runners scan the image without the PowerShell pipe/memory issues seen locally on Windows.

`docker compose ps` should show four containers healthy.
`http://localhost:8080` → kafka-ui.
