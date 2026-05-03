from fastapi import FastAPI

from click_rec.api.lifespan import lifespan
from click_rec.api.middleware import BodySizeLimitMiddleware
from click_rec.api.routers.events import router as events_router
from click_rec.api.routers.explain import router as explain_router
from click_rec.api.routers.items import router as items_router
from click_rec.api.routers.recommendations import router as recommendations_router
from click_rec.api.routers.search import router as search_router
from click_rec.config import get_settings
from click_rec.telemetry.http_metrics import instrument_http
from click_rec.telemetry.metrics import mount_metrics
from click_rec.telemetry.tracing import instrument_app

settings = get_settings()

app = FastAPI(title="SearchPulse", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    BodySizeLimitMiddleware,
    max_event_bytes=settings.max_event_bytes,
    max_batch_bytes=settings.max_batch_bytes,
)

app.include_router(events_router)
app.include_router(items_router)
app.include_router(search_router)
app.include_router(recommendations_router)
app.include_router(explain_router)

# Instrument at construction time, not in lifespan: FastAPI builds its
# middleware stack lazily on the first request, and `add_middleware`
# called from inside lifespan startup is fragile across Starlette/FastAPI
# versions. The OTEL FastAPIInstrumentor uses a ProxyTracer, so
# `init_tracing` (which sets the global provider) can still run later in
# lifespan and the middleware will pick it up at request time.
instrument_app(app)
instrument_http(app)
mount_metrics(app)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "env": settings.env}
