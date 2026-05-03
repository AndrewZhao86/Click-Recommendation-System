import os
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    kafka_bootstrap: str = "localhost:9094"
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = (
        "postgresql+asyncpg://clickrec:clickrec@localhost:5432/clickrec"
    )

    # Phase 7 — Gemini-only LLM layer.
    # `gemini_runtime_model` powers query-understanding, re-rank, /explain;
    # `gemini_judge_model` is reserved for the LLM-as-judge eval pass so a
    # bigger model grades the smaller one. Both default to free-tier IDs.
    # Per-use-case timeouts live in ranker.yaml under `llm:` (loaded by
    # `LLMConfig`), not here — secrets/model names belong in env, tunables
    # in YAML. See planning/phase7plan.md scope decision #9.
    gemini_api_key: str = ""
    gemini_runtime_model: str = "gemini-2.5-flash"
    gemini_judge_model: str = "gemini-2.5-pro"

    log_level: str = "INFO"
    env: str = "local"
    # Loaded by sentence-transformers in Phase 2 (catalogue embedding).
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    otel_exporter: Literal["stdout", "jaeger", "none"] = "stdout"
    # Phase 8b: Jaeger 1.62 accepts OTLP gRPC natively on 4317. The dev
    # docker-compose maps that port; on a host where it's taken the
    # operator overrides this to e.g. http://localhost:4318.
    otel_endpoint: str = "http://localhost:4317"
    # Default service name for the API process; the consumer overrides
    # this via OTEL_SERVICE_NAME so trace waterfalls split cleanly.
    otel_service_name: str = "searchpulse-api"
    prometheus_enabled: bool = True

    max_event_bytes: int = 1024
    max_batch_bytes: int = 500_000
    max_batch_events: int = 500
    dedupe_ttl_seconds: int = 86_400

    consumer_group: str = "click-enricher"
    consumer_max_retries: int = 3
    consumer_backoff_base_s: float = 0.5
    consumer_poll_timeout_ms: int = 1000
    consumer_max_records: int = 100
    recent_clicks_cap: int = 100
    recent_clicks_ttl_seconds: int = 86_400
    session_window: int = 5
    popularity_ttl_seconds: int = 3600
    item_cache_max_size: int = 10_000

    # Phase 5 — Redis hot-cache layer
    cache_item_ttl_seconds: int = 600
    cache_top_category_ttl_seconds: int = 300
    cache_lock_ttl_ms: int = 2000
    cache_lock_wait_poll_ms: int = 20
    cache_lock_wait_max_ms: int = 1500
    cache_negative_ttl_seconds: int = 60
    cache_refresh_interval_seconds: int = 60
    cache_refresh_half_life_seconds: int = 3600
    cache_top_max_members: int = 200

    # Phase 8 — query-embedding cache (in front of SentenceTransformer).
    # Sized at 1 day: query semantics don't drift in shorter windows and
    # the model name in the key prefix invalidates on swap.
    query_embedding_ttl_seconds: int = 86_400

    # Phase 6 — hybrid ranker
    ranker_config_path: str = "ranker.yaml"
    ranker_enabled: bool = True
    eval_output_dir: str = "artifacts"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    # The google-genai SDK auto-detects `GOOGLE_API_KEY`; if the operator
    # configured `GEMINI_API_KEY` instead, mirror it into the env so a
    # downstream `genai.Client()` picks it up without an explicit kwarg.
    # Done here (not in `__init__`) so settings stays pure-data and the
    # side-effect runs once per cache-warm rather than on every read.
    if settings.gemini_api_key and not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = settings.gemini_api_key
    return settings
