from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="Smart Campus Library RAG", alias="APP_NAME")
    data_dir: Path = Field(default=Path("data/raw"), alias="DATA_DIR")
    processed_dir: Path = Field(default=Path("data/processed"), alias="PROCESSED_DIR")
    metadata_db_path: Path = Field(default=Path("data/processed/library_metadata.sqlite"), alias="METADATA_DB_PATH")
    milvus_uri: str = Field(default="", alias="MILVUS_URI")
    milvus_host: str = Field(default="localhost", alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")
    milvus_timeout: int = Field(default=180, alias="MILVUS_TIMEOUT")
    milvus_collection: str = Field(default="library_books", alias="MILVUS_COLLECTION")
    milvus_drop_old: bool = Field(default=False, alias="MILVUS_DROP_OLD")
    milvus_connect_timeout: int = Field(default=180, alias="MILVUS_CONNECT_TIMEOUT")
    api_key: str = Field(default="", alias="API_KEY")
    embedding_backend: str = Field(default="auto", alias="EMBEDDING_BACKEND")
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_base_url: str = Field(default="", alias="EMBEDDING_BASE_URL")
    embedding_dimensions: int = Field(default=1024, alias="EMBEDDING_DIMENSIONS")
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com/v1", alias="DEEPSEEK_BASE_URL")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")
    chat_history_size: int = Field(default=6, alias="CHAT_HISTORY_SIZE")
    search_top_k: int = Field(default=3, alias="SEARCH_TOP_K")
    rerank_top_k: int = Field(default=3, alias="RERANK_TOP_K")
    similarity_threshold: float = Field(default=0.65, alias="SIMILARITY_THRESHOLD")
    hybrid_bm25_weight: float = Field(default=0.45, alias="HYBRID_BM25_WEIGHT")
    hybrid_vector_weight: float = Field(default=0.55, alias="HYBRID_VECTOR_WEIGHT")
    chunk_max_size: int = Field(default=800, alias="CHUNK_MAX_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")

    @property
    def resolved_milvus_uri(self) -> str:
        if self.milvus_uri.strip():
            return self.milvus_uri.strip()
        return f"http://{self.milvus_host}:{self.milvus_port}"

    @property
    def resolved_milvus_timeout(self) -> int:
        if "milvus_timeout" in self.model_fields_set:
            return self.milvus_timeout
        return self.milvus_connect_timeout


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.metadata_db_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
