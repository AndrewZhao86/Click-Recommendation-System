"""OpenTelemetry SDK + auto-instrumentation bootstrap.

Two entry points:

- `init_tracing()` — set up the global tracer provider + exporter.
  Idempotent (safe under uvicorn `--reload`). Honours `OTEL_EXPORTER`:
    - `none` → no-op (CI / unit tests).
    - `stdout` → `ConsoleSpanExporter` (default for local dev).
    - `jaeger` → OTLP gRPC to `OTEL_ENDPOINT` (Jaeger 1.62 OTLP-native).
- `instrument_app(app)` — auto-instrument FastAPI + SQLAlchemy + Redis
  + aiokafka. Called from the FastAPI lifespan so request handlers and
  every downstream client share one trace.

The aiokafka instrumentation is gated behind a try/except so a missing
`opentelemetry-instrumentation-aiokafka` package degrades to "API +
Postgres + Redis spans only" rather than 5xx-ing on startup. Plan
risks: Phase 8a "AIOKafkaInstrumentor pulls in a contrib package".
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from click_rec.config import get_settings

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

_initialised = False


def init_tracing(service_name: str | None = None) -> None:
    """Configure the global tracer provider. Idempotent.

    `service_name` overrides `settings.otel_service_name` (the consumer
    process passes "searchpulse-consumer" so trace waterfalls split
    cleanly from the API).
    """
    global _initialised
    if _initialised:
        return

    settings = get_settings()
    if settings.otel_exporter == "none":
        _initialised = True
        return

    # Lazy imports so pytest collection / `--help` don't pay the OTEL
    # SDK import cost when the exporter is "none".
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resolved_name = service_name or settings.otel_service_name
    resource = Resource.create(
        {
            "service.name": resolved_name,
            "service.namespace": "searchpulse",
            "deployment.environment": settings.env,
        }
    )
    provider = TracerProvider(resource=resource)

    if settings.otel_exporter == "stdout":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif settings.otel_exporter == "jaeger":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
        except ImportError:
            logger.warning(
                "otlp grpc exporter unavailable, falling back to stdout",
            )
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        else:
            provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(endpoint=settings.otel_endpoint, insecure=True)
                )
            )

    trace.set_tracer_provider(provider)
    _initialised = True
    logger.info(
        "tracing initialised",
        extra={
            "exporter": settings.otel_exporter,
            "service": resolved_name,
        },
    )


def instrument_app(app: FastAPI) -> None:
    """Auto-instrument FastAPI + SQLAlchemy + Redis + aiokafka.

    Safe to call when `OTEL_EXPORTER=none` — instrumentors hang spans
    off whatever provider is registered (the default no-op provider
    when tracing is disabled), so request handlers don't pay any cost
    beyond the instrumentation no-op.
    """
    settings = get_settings()
    if settings.otel_exporter == "none":
        return

    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app)

    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        from click_rec.db.base import get_engine

        SQLAlchemyInstrumentor().instrument(engine=get_engine().sync_engine)
    except Exception:  # noqa: BLE001
        logger.exception("sqlalchemy instrumentation failed; continuing without it")

    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor

        RedisInstrumentor().instrument()
    except Exception:  # noqa: BLE001
        logger.exception("redis instrumentation failed; continuing without it")

    try:
        from opentelemetry.instrumentation.aiokafka import AIOKafkaInstrumentor

        AIOKafkaInstrumentor().instrument()
    except Exception:  # noqa: BLE001
        # Documented in phase8plan.md risks: contrib package may not
        # match the installed aiokafka version. Degrade silently.
        logger.warning(
            "aiokafka instrumentation unavailable — continuing without Kafka spans",
            exc_info=True,
        )


def reset_for_tests() -> None:
    """Clear the idempotency flag so tests can re-init."""
    global _initialised
    _initialised = False
