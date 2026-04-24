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

    llm_provider: Literal["gemini", "groq", "ollama"] = "gemini"
    llm_timeout_seconds: float = 2.0
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    log_level: str = "INFO"
    env: str = "local"
    # Loaded by sentence-transformers in Phase 2 (catalogue embedding).
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    otel_exporter: Literal["stdout", "jaeger", "none"] = "stdout"
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
