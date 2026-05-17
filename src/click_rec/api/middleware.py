"""Content-Length guard for ingestion endpoints.

Single-event POSTs cap at 1 KB; `/events/batch` caps at 500 KB. Checks are
header-only — Uvicorn already buffers request bodies, and trusting
Content-Length avoids a second full-body read just to count bytes. Non-
`/events/*` paths pass through unrestricted.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        *,
        max_event_bytes: int,
        max_batch_bytes: int,
    ) -> None:
        super().__init__(app)
        self._max_event_bytes = max_event_bytes
        self._max_batch_bytes = max_batch_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        if path.startswith("/events/"):
            limit = self._max_batch_bytes if path == "/events/batch" else self._max_event_bytes
            cl = request.headers.get("content-length")
            if cl is not None:
                try:
                    length = int(cl)
                except ValueError:
                    length = -1
                if length > limit:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "payload too large"},
                    )
        return await call_next(request)
