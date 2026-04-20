from fastapi import FastAPI

from click_rec.api.lifespan import lifespan
from click_rec.api.middleware import BodySizeLimitMiddleware
from click_rec.api.routers.events import router as events_router
from click_rec.config import get_settings

settings = get_settings()

app = FastAPI(title="SearchPulse", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    BodySizeLimitMiddleware,
    max_event_bytes=settings.max_event_bytes,
    max_batch_bytes=settings.max_batch_bytes,
)

app.include_router(events_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "env": settings.env}
