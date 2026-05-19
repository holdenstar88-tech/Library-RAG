from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str | None = None
    question: str = Field(min_length=1, max_length=2000)


class SourceItem(BaseModel):
    rank: int
    score: float
    title: str | None = None
    author: str | None = None
    isbn: str | None = None
    category: str | None = None
    source: str | None = None
    excerpt: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    confidence: float = 0.0
    fallback: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    status: str
    vector_store_ready: bool
    documents_loaded: int
    collection_name: str
    session_id: str = Field(default_factory=lambda: str(uuid4()))

