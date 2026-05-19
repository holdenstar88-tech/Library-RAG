from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.config import Settings, get_settings


@dataclass(frozen=True)
class EmbeddingProfile:
    backend: str
    model: str
    namespace: str


class VectorEmbeddingService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._backend = self._resolve_backend()
        self._profile = self._build_profile()
        self._embeddings = self._build_embeddings()

    def _shared_api_key(self) -> str:
        return self.settings.embedding_api_key or self.settings.api_key

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
        return slug or "default"

    def _resolve_backend(self) -> str:
        backend = (self.settings.embedding_backend or "auto").strip().lower()
        if backend not in {"auto", "openai", "dashscope", "huggingface"}:
            raise ValueError("EMBEDDING_BACKEND must be one of: auto, openai, dashscope, huggingface")
        if backend == "auto":
            shared_api_key = self._shared_api_key()
            if shared_api_key or self.settings.embedding_base_url:
                if not (shared_api_key and self.settings.embedding_base_url):
                    raise ValueError(
                        "Remote embedding requires both EMBEDDING_API_KEY or API_KEY and EMBEDDING_BASE_URL, "
                        "or set EMBEDDING_BACKEND=huggingface explicitly."
                    )
                if "dashscope.aliyuncs.com" in self.settings.embedding_base_url:
                    return "dashscope"
                return "openai"
            return "huggingface"
        if backend == "openai" and not (self._shared_api_key() and self.settings.embedding_base_url):
            raise ValueError("EMBEDDING_BACKEND=openai requires EMBEDDING_API_KEY or API_KEY and EMBEDDING_BASE_URL")
        if backend == "dashscope" and not self._shared_api_key():
            raise ValueError("EMBEDDING_BACKEND=dashscope requires EMBEDDING_API_KEY or API_KEY")
        return backend

    def _build_profile(self) -> EmbeddingProfile:
        if self._backend in {"openai", "dashscope"}:
            provider = self._slugify(self.settings.embedding_base_url.split("//", 1)[-1]) if self.settings.embedding_base_url else self._backend
            namespace = "_".join(
                part
                for part in [
                    self._backend,
                    provider,
                    self._slugify(self.settings.embedding_model),
                    f"d{self.settings.embedding_dimensions}" if self._backend == "openai" else "",
                ]
                if part
            )
            return EmbeddingProfile(backend=self._backend, model=self.settings.embedding_model, namespace=namespace)
        namespace = "_".join(
            part
            for part in [
                "huggingface",
                self._slugify(self.settings.embedding_model),
                self._slugify(self.settings.embedding_device),
            ]
            if part
        )
        return EmbeddingProfile(backend=self._backend, model=self.settings.embedding_model, namespace=namespace)

    def _build_embeddings(self) -> Embeddings:
        if self._backend == "openai":
            api_key = self._shared_api_key()
            return OpenAIEmbeddings(
                api_key=api_key,
                base_url=self.settings.embedding_base_url,
                model=self.settings.embedding_model,
                dimensions=self.settings.embedding_dimensions,
            )
        if self._backend == "dashscope":
            return DashScopeEmbeddings(
                model=self.settings.embedding_model,
                dashscope_api_key=self._shared_api_key(),
            )
        return HuggingFaceEmbeddings(
            model_name=self.settings.embedding_model,
            model_kwargs={"device": self.settings.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    @property
    def collection_namespace(self) -> str:
        return self._profile.namespace


@lru_cache(maxsize=1)
def get_vector_embedding_service() -> VectorEmbeddingService:
    return VectorEmbeddingService()
