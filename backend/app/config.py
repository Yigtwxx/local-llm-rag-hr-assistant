"""Application settings, loaded from environment variables.

Nothing here is hardcoded to a machine or a secret: every value can be
overridden through the environment or a local `.env` file.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    """Runtime configuration for the RAG backend."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama runtime
    ollama_host: str = "http://localhost:11434"
    ollama_timeout: float = 120.0

    # Models
    chat_model_primary: str = "qwen3.5:9b"
    chat_model_secondary: str = "gemma4:12b"
    embedding_model: str = "qwen3-embedding:0.6b"

    # Retrieval
    kb_dir: Path = PROJECT_ROOT / "data" / "kb"
    storage_dir: Path = BACKEND_ROOT / "storage"
    collection_name: str = "hr_kb"
    chunk_size: int = 500
    chunk_overlap: int = 75
    top_k: int = 4
    # Calibrated with `bench/calibrate_threshold.py`. An earlier value of 0.52
    # came from the benchmark's long, well-formed questions alone and rejected
    # short ones ("Harcırah ne kadar?", scoring 0.468) whose answers are in the
    # documents — 4 of 19 in-scope questions refused. With both phrasing styles
    # in the set the two groups overlap, so no threshold is error-free and the
    # choice is which error to buy. 0.46 is the highest value that misses no
    # answerable question; the out-of-scope questions that clear it are caught
    # by the system prompt, which is the second, independent defence.
    similarity_threshold: float = 0.46

    # Generation
    temperature: float = 0.0
    seed: int = 42
    # Hard cap on generated tokens. Sent on every call so a single verbose
    # answer cannot skew benchmark averages or stall the UI.
    max_tokens: int = 1024

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173"

    @property
    def chat_models(self) -> list[str]:
        """Both benchmarked chat models, primary first."""
        return [self.chat_model_primary, self.chat_model_secondary]

    @property
    def cors_origin_list(self) -> list[str]:
        return [
            origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
        ]

    def resolve_kb_dir(self) -> Path:
        """Absolute knowledge-base path, tolerating relative env values."""
        return (
            self.kb_dir
            if self.kb_dir.is_absolute()
            else (BACKEND_ROOT / self.kb_dir).resolve()
        )

    def resolve_storage_dir(self) -> Path:
        """Absolute vector-store path, tolerating relative env values."""
        path = (
            self.storage_dir
            if self.storage_dir.is_absolute()
            else (BACKEND_ROOT / self.storage_dir).resolve()
        )
        return path


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance, so the env file is parsed only once."""
    return Settings()
