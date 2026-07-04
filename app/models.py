from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str | None = None
    question: str = Field(min_length=1, max_length=2000)


class CatalogSearchRequest(BaseModel):
    query: str = ""
    category: str | None = None
    author: str | None = None
    title: str | None = None
    isbn: str | None = None
    book_id: str | None = None
    call_number: str | None = None
    main_character: str | None = None
    subject: str | None = None
    available_only: bool = False
    page: int = Field(default=1, ge=1)
    limit: int = Field(default=20, ge=1, le=50)


class SourceItem(BaseModel):
    rank: int
    score: float
    book_id: str | None = None
    title: str | None = None
    author: str | None = None
    isbn: str | None = None
    call_number: str | None = None
    category: str | None = None
    subjects: str | None = None
    main_characters: str | None = None
    plot_summary: str | None = None
    shelf: str | None = None
    shelf_code: str | None = None
    shelf_row: int | None = None
    shelf_col: int | None = None
    floor: str | None = None
    area: str | None = None
    copy_count: int | None = None
    available_count: int | None = None
    availability: str | None = None
    borrow_rule: str | None = None
    open_time: str | None = None
    source: str | None = None
    excerpt: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CatalogSearchResponse(BaseModel):
    query: str
    results: list[SourceItem] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    limit: int = 20
    total_pages: int = 0
    has_prev: bool = False
    has_next: bool = False
    categories: list[str] = Field(default_factory=list)
    fallback: bool = False


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
