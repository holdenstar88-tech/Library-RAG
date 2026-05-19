from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.services.library_rag import get_service


settings = get_settings()
app = FastAPI(title=settings.app_name)
BASE_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = BASE_DIR / "app" / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def startup() -> None:
    get_service()


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return get_service().health()


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    return get_service().answer(payload.question, payload.session_id)


@app.post("/api/reindex", response_model=HealthResponse)
def reindex() -> HealthResponse:
    service = get_service()
    service.sync()
    return service.health()
