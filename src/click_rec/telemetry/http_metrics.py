"""FastAPI HTTP request metrics via `prometheus-fastapi-instrumentator`.

OTEL alone won't expose `http_request_duration_seconds` on the
Prometheus `/metrics` endpoint without an OTEL→Prom bridge. The
instrumentator library handles it natively against the same
`prometheus_client.REGISTRY` that `mount_metrics()` already serves.

Series it adds (default config):
- `http_requests_total{method,handler,status}`
- `http_request_duration_seconds{handler}`

`/metrics` and `/health` are excluded from instrumentation so the
metrics endpoint doesn't appear in its own histograms (would skew
p95s) and so health-check pings don't pollute the request series.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from click_rec.config import get_settings

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


def instrument_http(app: FastAPI) -> None:
    """Attach the Prometheus FastAPI instrumentator to `app`.

    No-op when `prometheus_enabled=False` so unit tests building a bare
    FastAPI app don't accidentally register handlers on the global
    registry. `mount_metrics()` already owns the `/metrics` route, so
    we instrument-only and skip the library's `expose()` call.
    """
    if not get_settings().prometheus_enabled:
        return

    try:
        from prometheus_fastapi_instrumentator import Instrumentator
    except ImportError:
        logger.warning(
            "prometheus-fastapi-instrumentator missing — skipping HTTP metrics",
        )
        return

    Instrumentator(
        should_group_status_codes=False,
        excluded_handlers=["/metrics", "/health"],
    ).instrument(app)
