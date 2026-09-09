"""Centralized application configuration for LexisGraph.

Uses Pydantic Settings (v2) to parse environment variables from `.env`
and the host environment, providing type-safe settings and sensible defaults.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """LexisGraph application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --------------------------------------------------------------------------
    # Application Settings
    # --------------------------------------------------------------------------
    app_name: str = Field(default="LexisGraph", description="Application name")
    app_env: Literal["development", "staging", "production", "test"] = Field(
        default="development", description="Current operating environment"
    )
    debug: bool = Field(default=True, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Application logging level")
    api_v1_prefix: str = Field(default="/api/v1", description="Prefix for API v1 routes")
    host: str = Field(default="0.0.0.0", description="Host address for FastAPI server")
    port: int = Field(default=8000, ge=1, le=65535, description="Port number for FastAPI server")

    # --------------------------------------------------------------------------
    # Primary LLM Provider: Groq
    # --------------------------------------------------------------------------
    groq_api_key: str = Field(default="", description="Groq API key")
    groq_model: str = Field(
        default="llama-3.3-70b-versatile", description="Groq model for reasoning and generation"
    )
    groq_vision_model: str = Field(
        default="llama-3.2-11b-vision-preview",
        description="Groq vision model for visual text extraction",
    )
    groq_temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Temperature for Groq LLM generations"
    )
    groq_max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum retry attempts on Groq API errors"
    )

    # --------------------------------------------------------------------------
    # Optional Fallback LLM: OpenAI
    # --------------------------------------------------------------------------
    openai_api_key: str = Field(default="", description="OpenAI API key for fallback")
    openai_model: str = Field(default="gpt-4o", description="OpenAI fallback model")

    # --------------------------------------------------------------------------
    # Vector Database: Qdrant
    # --------------------------------------------------------------------------
    qdrant_host: str = Field(default="localhost", description="Qdrant host address")
    qdrant_port: int = Field(default=6333, ge=1, le=65535, description="Qdrant HTTP port")
    qdrant_grpc_port: int = Field(default=6334, ge=1, le=65535, description="Qdrant gRPC port")
    qdrant_api_key: str = Field(default="", description="Qdrant API key (optional)")
    qdrant_collection_name: str = Field(
        default="lexisgraph_legal_documents", description="Name of multi-vector collection"
    )
    qdrant_vector_dim: int = Field(
        default=128, ge=1, description="Embedding dimension for ColPali token multi-vectors"
    )
    qdrant_use_https: bool = Field(default=False, description="Use HTTPS for Qdrant connection")

    # --------------------------------------------------------------------------
    # Knowledge Graph: Neo4j
    # --------------------------------------------------------------------------
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j Bolt connection URI")
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="legalgraphpassword123", description="Neo4j password")
    neo4j_database: str = Field(default="neo4j", description="Neo4j default database name")

    # --------------------------------------------------------------------------
    # ColPali & Embedding Configuration
    # --------------------------------------------------------------------------
    colpali_model_name: str = Field(
        default="vidore/colpali-v1.3-hf", description="ColPali model repository on Hugging Face"
    )
    colpali_device: str = Field(
        default="auto", description="Device for ColPali inference ('auto', 'cuda', 'cpu')"
    )
    colpali_batch_size: int = Field(
        default=4, ge=1, description="Batch size for page image embeddings"
    )
    colpali_dpi: int = Field(
        default=200, ge=72, le=600, description="DPI for converting PDF pages to images"
    )
    poppler_path: str | None = Field(
        default=None, description="Path to poppler binaries directory (optional)"
    )

    # --------------------------------------------------------------------------
    # Security & Presidio PII Masking
    # --------------------------------------------------------------------------
    presidio_enabled: bool = Field(default=True, description="Enable Presidio PII detection")
    presidio_default_lang: str = Field(default="ar", description="Default language for PII analysis")
    presidio_spacy_model: str = Field(
        default="xx_ent_wiki_sm", description="spaCy model for multilingual NER"
    )
    presidio_masking_char: str = Field(default="*", description="Character used for masking PII")
    nemo_guardrails_enabled: bool = Field(default=True, description="Enable NeMo safety rails")

    # --------------------------------------------------------------------------
    # BM25 Keyword Search & Fusion
    # --------------------------------------------------------------------------
    bm25_top_k: int = Field(default=10, ge=1, description="Top-k documents retrieved by BM25")
    rrf_k_constant: int = Field(
        default=60, ge=1, description="Smoothing constant (k) for Reciprocal Rank Fusion"
    )

    # --------------------------------------------------------------------------
    # Document Storage & AWS (Production)
    # --------------------------------------------------------------------------
    storage_mode: Literal["local", "s3"] = Field(
        default="local", description="Storage backend ('local' or 's3')"
    )
    local_storage_dir: str = Field(
        default="./data/documents", description="Path for local document storage"
    )
    aws_region: str = Field(default="us-east-1", description="AWS default region")
    s3_bucket_name: str = Field(
        default="lexisgraph-legal-docs", description="S3 bucket for document storage"
    )
    aws_access_key_id: str = Field(default="", description="AWS Access Key ID")
    aws_secret_access_key: str = Field(default="", description="AWS Secret Access Key")

    # --------------------------------------------------------------------------
    # Computed Properties & Helpers
    # --------------------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        """Check if currently running in production environment."""
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        """Check if currently running in development environment."""
        return self.app_env == "development"

    @property
    def qdrant_url(self) -> str:
        """Construct full HTTP/HTTPS URL for Qdrant client."""
        protocol = "https" if self.qdrant_use_https else "http"
        return f"{protocol}://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def resolved_device(self) -> str:
        """Resolve embedding device dynamically if set to 'auto'."""
        if self.colpali_device != "auto":
            return self.colpali_device
        import shutil
        import subprocess
        import sys

        # If no nvidia-smi tool exists on PATH, CUDA is unavailable
        if not shutil.which("nvidia-smi"):
            return "cpu"

        try:
            res = subprocess.run(
                [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0 and res.stdout.strip() == "True":
                return "cuda"
        except Exception:
            pass
        return "cpu"

    @property
    def has_groq(self) -> bool:
        """Check if Groq API key is configured."""
        return bool(self.groq_api_key.strip())

    @property
    def has_openai(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.openai_api_key.strip())

    @property
    def storage_path(self) -> Path:
        """Get Path object for local storage directory."""
        return Path(self.local_storage_dir)


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
