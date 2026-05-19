"""Round-robin reverse proxy for the Windows multi-worker workaround.

`uvicorn --workers N` is unreliable on Windows (socket-inheritance race —
see planning/phase8planextra.md §5). The workaround is to run N separate
single-worker uvicorn processes on different ports and round-robin in
front of them. This script is that round-robin.

Defaults: forward to 127.0.0.1:8001..8004, listen on 0.0.0.0:8000.
Override with $env:BACKENDS / $env:PROXY_PORT.

    PS> uv run python scripts/proxy.py
    PS> $env:BACKENDS="127.0.0.1:8001,127.0.0.1:8002"; uv run python scripts/proxy.py

Run AFTER all backend uvicorns are up; verify with:
    netstat -ano | Select-String ":800[0-4].*LISTENING"
"""

from __future__ import annotations

import itertools
import os

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

_DEFAULT_BACKENDS = "127.0.0.1:8001,127.0.0.1:8002,127.0.0.1:8003,127.0.0.1:8004"
BACKENDS = [
    b.strip()
    for b in os.environ.get("BACKENDS", _DEFAULT_BACKENDS).split(",")
    if b.strip()
]
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8000"))

# Hop-by-hop headers must not be forwarded — RFC 7230 §6.1.
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

_rr = itertools.cycle(BACKENDS)
_client: httpx.AsyncClient | None = None


def _next_backend() -> str:
    return next(_rr)


async def _proxy(request: Request) -> Response:
    assert _client is not None
    backend = _next_backend()
    url = f"http://{backend}{request.url.path}"
    if request.url.query:
        url += f"?{request.url.query}"

    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
    body = await request.body()

    try:
        upstream = await _client.request(
            request.method,
            url,
            headers=headers,
            content=body,
        )
    except httpx.RequestError as exc:
        return Response(f"upstream {backend} unreachable: {exc}", status_code=502)

    out_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_BY_HOP}
    return Response(content=upstream.content, status_code=upstream.status_code, headers=out_headers)


async def _startup() -> None:
    global _client
    # Generous pool: 4 backends × expected concurrent in-flight from locust.
    limits = httpx.Limits(max_connections=400, max_keepalive_connections=200)
    _client = httpx.AsyncClient(timeout=httpx.Timeout(60.0), limits=limits)
    print(f"proxy: listening on 0.0.0.0:{PROXY_PORT}, round-robin over {BACKENDS}", flush=True)


async def _shutdown() -> None:
    if _client is not None:
        await _client.aclose()


_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
app = Starlette(
    routes=[Route("/{path:path}", _proxy, methods=_METHODS)],
    on_startup=[_startup],
    on_shutdown=[_shutdown],
)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, log_level="warning")
