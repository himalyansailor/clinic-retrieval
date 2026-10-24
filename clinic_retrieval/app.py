"""The API. Run it with:

    uvicorn app:app --reload

Then open http://127.0.0.1:8000 for the client, or /docs for the schema.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from clinic_retrieval.deps import get_index
from clinic_retrieval.routers import analytics, ask, health

STATIC = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_index()          # build the index before the first request, not during
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Clinic retrieval",
    description="Health schemes, drug availability and clinic protocols, "
                "in Hindi and English, answered offline.",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(ask.router)
app.include_router(analytics.router)
app.mount("/ui", StaticFiles(directory=STATIC, html=True), name="ui")


@app.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    return RedirectResponse("/ui/")
