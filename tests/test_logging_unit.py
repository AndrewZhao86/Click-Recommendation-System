"""Unit tests for `telemetry.logging.configure_logging`.

Validates JSON output and OTEL trace_id correlation. Each test resets
the configurator so the global handler set is fresh.
"""

from __future__ import annotations

import json
import logging
from io import StringIO

import pytest

from click_rec.telemetry import logging as telemetry_logging


@pytest.fixture(autouse=True)
def _reset_logging() -> None:
    telemetry_logging.reset_for_tests()
    yield
    telemetry_logging.reset_for_tests()


def _capture_one_record(level: str = "INFO") -> dict[str, object]:
    """Configure logging, point handler at a buffer, emit one record."""
    telemetry_logging.configure_logging(level)
    root = logging.getLogger()
    # Replace the configured StreamHandler's stream with a buffer so we
    # can read what was written.
    handler = root.handlers[0]
    buf = StringIO()
    handler.stream = buf  # type: ignore[attr-defined]

    logger = logging.getLogger("test_logger")
    logger.info("hello", extra={"foo": "bar"})

    raw = buf.getvalue().strip()
    assert raw, "handler emitted nothing"
    return json.loads(raw)


def test_configure_logging_emits_json() -> None:
    record = _capture_one_record()
    # JSON shape: timestamp, level, event message, plus extras.
    assert record["event"] == "hello"
    assert record["level"] == "info"
    assert "timestamp" in record


def test_configure_logging_includes_extras() -> None:
    record = _capture_one_record()
    assert record["foo"] == "bar"


def test_log_record_no_trace_id_when_no_span() -> None:
    """No active span → no `trace_id` field on the record."""
    record = _capture_one_record()
    assert "trace_id" not in record


def test_configure_logging_idempotent() -> None:
    telemetry_logging.configure_logging("INFO")
    handlers_first = list(logging.getLogger().handlers)
    telemetry_logging.configure_logging("INFO")
    handlers_second = list(logging.getLogger().handlers)
    # Idempotent: no new handlers added on second call.
    assert handlers_second == handlers_first


def test_log_record_includes_trace_id_when_span_active() -> None:
    """An active OTEL span attaches `trace_id` + `span_id` to the record."""
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    # Stand up a tracer provider so spans have valid contexts.
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer("test")

    telemetry_logging.configure_logging("INFO")
    root = logging.getLogger()
    handler = root.handlers[0]
    buf = StringIO()
    handler.stream = buf  # type: ignore[attr-defined]

    logger = logging.getLogger("test_logger")
    with tracer.start_as_current_span("test-span"):
        logger.info("inside-span")

    raw = buf.getvalue().strip()
    record = json.loads(raw)
    assert "trace_id" in record
    assert len(record["trace_id"]) == 32
    assert "span_id" in record
    assert len(record["span_id"]) == 16
