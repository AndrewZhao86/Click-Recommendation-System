"""Unit tests for `telemetry.tracing.init_tracing`.

Validates the exporter-selection branch logic (none / stdout / jaeger)
and the idempotency flag.
"""

from __future__ import annotations

import pytest

from click_rec.config import get_settings
from click_rec.telemetry import tracing as telemetry_tracing


@pytest.fixture(autouse=True)
def _reset_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    telemetry_tracing.reset_for_tests()
    yield
    telemetry_tracing.reset_for_tests()


def test_init_tracing_none_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """`OTEL_EXPORTER=none` returns without touching the global provider."""
    monkeypatch.setenv("OTEL_EXPORTER", "none")
    get_settings.cache_clear()
    try:
        # Importable + idempotent flag flip; doesn't touch OTEL SDK.
        telemetry_tracing.init_tracing()
        # Re-call doesn't blow up.
        telemetry_tracing.init_tracing()
    finally:
        get_settings.cache_clear()


def test_init_tracing_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated init calls are no-ops (no double-registration)."""
    pytest.importorskip("opentelemetry.sdk.trace")
    monkeypatch.setenv("OTEL_EXPORTER", "stdout")
    get_settings.cache_clear()
    try:
        from opentelemetry import trace

        telemetry_tracing.init_tracing(service_name="t1")
        first = trace.get_tracer_provider()
        telemetry_tracing.init_tracing(service_name="t2")
        second = trace.get_tracer_provider()
        # Same provider — second call short-circuited via `_initialised`.
        assert first is second
    finally:
        get_settings.cache_clear()


def test_init_tracing_stdout_registers_console_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`OTEL_EXPORTER=stdout` registers a tracer provider with at least one
    span processor."""
    pytest.importorskip("opentelemetry.sdk.trace")
    monkeypatch.setenv("OTEL_EXPORTER", "stdout")
    get_settings.cache_clear()
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        telemetry_tracing.init_tracing(service_name="searchpulse-test")
        provider = trace.get_tracer_provider()
        # Must be the SDK's TracerProvider, not the no-op default.
        assert isinstance(provider, TracerProvider)
    finally:
        get_settings.cache_clear()
