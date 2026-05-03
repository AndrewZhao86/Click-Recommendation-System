"""structlog JSON logging + OTEL `trace_id` correlation.

`configure_logging()` is the single configurator used by both
`api/lifespan.py` and `cli.py`. It wires:

1. A stdlib root logger whose handler uses structlog's
   `ProcessorFormatter` to render every record as JSON. This means
   *existing* `logging.getLogger(__name__).info("foo", extra=...)` call
   sites everywhere in the codebase silently gain JSON output —
   without rewrites.
2. structlog's processor chain so `structlog.get_logger()` calls (none
   today, but reserved for new sites) get the same JSON shape.
3. A `_add_trace_id` processor that reads the OTEL active span and
   attaches `trace_id` + `span_id` fields when a span is in scope.

Idempotent — repeated calls are a no-op so reload-mode uvicorn doesn't
double-attach handlers.
"""

from __future__ import annotations

import logging
from typing import Any

import structlog

_configured = False


def _add_trace_id(_: Any, __: Any, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Attach trace_id / span_id from the OTEL active span, if any.

    Cheap when no provider is registered: `get_current_span()` returns
    the no-op span, whose context is invalid and we skip the attach.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context() if span is not None else None
        if ctx is not None and ctx.is_valid:
            event_dict["trace_id"] = format(ctx.trace_id, "032x")
            event_dict["span_id"] = format(ctx.span_id, "016x")
    except Exception:  # noqa: BLE001
        # OTEL may not be installed in some test envs — never fail the
        # log call because of an instrumentation hiccup.
        pass
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    """Configure stdlib + structlog to emit JSON with trace correlation.

    Safe to call multiple times — repeated invocations short-circuit.
    """
    global _configured
    if _configured:
        return

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_trace_id,
    ]

    # Render JSON at the very end. ExceptionPrettyPrinter would dump a
    # multi-line traceback that breaks JSON-per-line tooling; the
    # default `format_exc_info` keeps the traceback as a single field.
    json_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors + [
            structlog.stdlib.ExtraAdder(),
        ],
        processors=[
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(json_formatter)

    root = logging.getLogger()
    # Replace whatever the framework set up first so JSON is the only
    # handler — otherwise uvicorn's default text handler keeps emitting
    # plain lines alongside our JSON.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())

    structlog.configure(
        processors=shared_processors + [
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _configured = True


def reset_for_tests() -> None:
    """Tear down so a test can re-configure with a different level."""
    global _configured
    _configured = False
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
