"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from backend.api.upload import router as upload_router
from backend.config import get_settings
from backend.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and clean up application resources."""

    settings = get_settings()
    settings.create_data_directories()

    yield


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
    description=(
        "Human-in-the-loop regulatory policy gap analysis API. "
        "This system does not provide legal advice."
    ),
)

app.include_router(upload_router)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
)
async def health_check() -> HealthResponse:
    """Return basic application health information."""

    return HealthResponse(
        application=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
    )