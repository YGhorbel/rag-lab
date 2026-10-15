"""Central configuration for RAG Lab (Phase 0 defaults)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    root_dir: Path = ROOT_DIR
    data_raw_dir: Path = ROOT_DIR / "data" / "raw"
    data_processed_dir: Path = ROOT_DIR / "data" / "processed"
    data_benchmarks_dir: Path = ROOT_DIR / "data" / "benchmarks"
    experiments_dir: Path = ROOT_DIR / "experiments"

    catalog_version: str = "v1"
    dataset_seed: int = 42
    catalog_size: int = 2000
    benchmark_size: int = 500
    demo_set_size: int = 80

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "products_v1"
    ollama_base_url: str = "https://ollama.com"
    ollama_api_key: str = ""

    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    generator_model: str = "gpt-oss:20b"

    dense_top_k: int = 50
    lexical_top_k: int = 50
    rerank_top_k: int = 10
    context_budget_chars: int = 6000

    streamlit_port: int = Field(default=8501, alias="STREAMLIT_SERVER_PORT")

    @property
    def catalog_path(self) -> Path:
        return self.data_processed_dir / f"catalog_{self.catalog_version}.jsonl"

    @property
    def catalog_meta_path(self) -> Path:
        return self.data_processed_dir / f"catalog_{self.catalog_version}.meta.json"

    @property
    def benchmark_path(self) -> Path:
        return self.data_benchmarks_dir / f"benchmark_{self.catalog_version}.jsonl"

    @property
    def demo_benchmark_path(self) -> Path:
        return self.data_benchmarks_dir / f"demo_{self.catalog_version}.jsonl"


def get_settings() -> Settings:
    return Settings()
