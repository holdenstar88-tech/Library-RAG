from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from app.retrieval.hybrid import RetrievedChunk
from app.services.vector_embedding_service import get_vector_embedding_service
from app.services.vector_store_manager import get_vector_store_manager


class VectorSearchService:
    def __init__(self) -> None:
        self.embeddings = get_vector_embedding_service().embeddings
        self.store_manager = get_vector_store_manager()

    def search_similar_documents(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        try:
            scored = self.store_manager.similarity_search_with_score(query, top_k)
        except Exception:
            return []
        results: list[RetrievedChunk] = []
        for rank, (document, score) in enumerate(scored, start=1):
            results.append(
                RetrievedChunk(
                    document=document,
                    score=float(score),
                    source="vector",
                    rank=rank,
                )
            )
        return results


@lru_cache(maxsize=1)
def get_vector_search_service() -> VectorSearchService:
    return VectorSearchService()
