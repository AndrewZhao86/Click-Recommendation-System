# SearchPulse

Real-time, personalised product-search & recommendation engine.
Kafka-backed click ingestion, Redis hot-cache, hybrid ranker (BM25 + pgvector + co-click), optional LLM re-rank.

See [planning/plan.md](planning/plan.md) for the full design and phase breakdown.

## Quickstart

```bash
cp .env.example .env
make install        # uv sync --extra dev
make up             # kafka (KRaft) + kafka-ui + redis + postgres (pgvector)
make test           # smoke: GET /health
make api            # uvicorn on :8000
```

`docker compose ps` should show four containers healthy.
`http://localhost:8080` → kafka-ui.
