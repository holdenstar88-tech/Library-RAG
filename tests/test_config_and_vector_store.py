from __future__ import annotations

from app.config import Settings
from app.services.vector_store_manager import VectorStoreManager


def test_settings_resolve_milvus_uri_and_timeout(monkeypatch):
    monkeypatch.setenv("MILVUS_URI", "")
    monkeypatch.setenv("MILVUS_HOST", "milvus")
    monkeypatch.setenv("MILVUS_PORT", "19530")
    monkeypatch.setenv("MILVUS_TIMEOUT", "42")

    settings = Settings(
        MILVUS_URI="",
        MILVUS_HOST="milvus",
        MILVUS_PORT=19530,
        MILVUS_TIMEOUT=42,
    )

    assert settings.resolved_milvus_uri == "http://milvus:19530"
    assert settings.resolved_milvus_timeout == 42


def test_vector_store_collection_is_namespaced(monkeypatch):
    class DummyEmbeddingService:
        collection_namespace = "dashscope_text_embedding_v4"

    monkeypatch.setattr(
        "app.services.vector_store_manager.get_vector_embedding_service",
        lambda: DummyEmbeddingService(),
    )

    settings = Settings(milvus_collection="library_books")
    manager = VectorStoreManager(settings=settings)

    assert manager.resolved_collection_name == "library_books__dashscope_text_embedding_v4"
