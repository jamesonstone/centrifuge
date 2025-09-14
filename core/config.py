"""Configuration management for Centrifuge."""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra environment variables instead of forbidding them
    )

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/centrifuge"

    # MinIO/S3
    artifact_endpoint: str = "http://localhost:9000"
    artifact_bucket: str = "centrifuge"
    artifact_access_key: str = "minioadmin"
    artifact_secret_key: str = "minioadmin"
    artifact_region: str = "us-east-1"

    # LLM Configuration (Direct Library Mode)
    # Note: llm_base_url and llm_provider kept for backward compatibility but unused in direct mode
    llm_base_url: str = "http://localhost:4000"  # Deprecated - for proxy mode only
    llm_model_id: str = "openai/gpt-5"
    llm_provider: str = "direct"  # Changed from "litellm" to indicate direct library usage
    llm_temperature: float = 0.0
    llm_seed: int = 42
    openai_api_key: Optional[str] = None

    # Additional LLM providers
    google_genai_use_vertexai: bool = False
    gemini_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Processing limits
    edit_cap_pct: int = 20
    confidence_floor: float = 0.80
    row_limit: int = 50000
    batch_size: int = 15  # for LLM batching

    # API settings
    max_file_size: int = 50 * 1024 * 1024  # 50MB

    # Cache settings
    cache_ttl_seconds: int = 86400  # 24 hours
    cache_max_size: int = 10000

    # MinIO settings (aliases for compatibility)
    @property
    def MINIO_ENDPOINT(self) -> str:
        return self.artifact_endpoint

    @property
    def MINIO_BUCKET(self) -> str:
        return self.artifact_bucket

    @property
    def MAX_FILE_SIZE(self) -> int:
        return self.max_file_size

    # Worker configuration
    worker_id: str = "worker-default"
    heartbeat_interval: int = 30  # seconds
    visibility_timeout: int = 300  # seconds

    # Development
    environment: str = "development"
    debug: bool = False
    mock_llm: bool = False  # Use mock LLM in dev mode

    def validate_production(self) -> None:
        """Validate required settings for production mode."""
        if self.environment == "production":
            if not self.openai_api_key and not self.mock_llm:
                raise ValueError(
                    "OPENAI_API_KEY is required in production mode. "
                    "Set MOCK_LLM=true for development without API key."
                )

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"


# Global settings instance
settings = Settings()

# Validate on import
try:
    settings.validate_production()
except ValueError as e:
    if settings.is_production:
        raise
    else:
        # Log warning in development but don't fail
        print(f"Warning: {e}")
