"""API request and response models."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from core.models import RunState


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")


class RunSubmitRequest(BaseModel):
    """Run submission request."""
    s3_url: Optional[str] = Field(None, description="S3 URL to input CSV")
    schema_version: str = Field("1.0.0", description="Schema version to use")
    use_inferred: bool = Field(False, description="Enable schema inference (experimental)")
    dry_run: bool = Field(False, description="Validate without applying changes")
    llm_columns: Optional[List[str]] = Field(
        None,
        description="Columns to process with LLM (default: department, account_name)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "s3_url": "s3://bucket/path/to/data.csv",
                "schema_version": "1.0.0",
                "use_inferred": False,
                "dry_run": False,
                "llm_columns": ["department", "account_name"]
            }
        }


class RunSubmitResponse(BaseModel):
    """Run submission response."""
    run_id: UUID = Field(..., description="Unique run identifier")
    state: RunState = Field(..., description="Current run state")
    message: str = Field(..., description="Status message")
    estimated_duration_seconds: int = Field(..., description="Estimated processing time")
    
    class Config:
        schema_extra = {
            "example": {
                "run_id": "123e4567-e89b-12d3-a456-426614174000",
                "state": "queued",
                "message": "Run queued for processing",
                "estimated_duration_seconds": 30
            }
        }


class RunArtifact(BaseModel):
    """Artifact metadata."""
    name: str = Field(..., description="Artifact name")
    size_bytes: int = Field(..., description="File size in bytes")
    content_hash: str = Field(..., description="SHA256 hash of content")
    download_url: str = Field(..., description="URL to download artifact")


class RunStatusResponse(BaseModel):
    """Run status response."""
    run_id: UUID = Field(..., description="Unique run identifier")
    state: RunState = Field(..., description="Current run state")
    created_at: str = Field(..., description="Run creation timestamp")
    phase_progress: Dict[str, int] = Field(
        default_factory=dict,
        description="Progress percentage by phase"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Run metrics summary"
    )
    artifacts: List[RunArtifact] = Field(
        default_factory=list,
        description="Available artifacts"
    )
    error: Optional[str] = Field(None, description="Error message if failed")
    
    class Config:
        schema_extra = {
            "example": {
                "run_id": "123e4567-e89b-12d3-a456-426614174000",
                "state": "succeeded",
                "created_at": "2024-01-01T12:00:00Z",
                "phase_progress": {
                    "ingest": 100,
                    "validation_1": 100,
                    "rules_engine": 100,
                    "residual_planner": 100,
                    "llm_adapter": 100,
                    "apply_patches": 100,
                    "validation_2": 100,
                    "artifacts": 100
                },
                "metrics": {
                    "total_rows": 1000,
                    "clean_rows": 950,
                    "quarantined_rows": 50,
                    "success_rate": 0.95,
                    "rules_fixed_count": 800,
                    "llm_fixed_count": 150
                },
                "artifacts": [
                    {
                        "name": "cleaned.csv",
                        "size_bytes": 102400,
                        "content_hash": "abc123...",
                        "download_url": "/runs/123e4567.../artifacts/cleaned.csv"
                    }
                ],
                "error": None
            }
        }


class RunListItem(BaseModel):
    """Run list item."""
    run_id: str = Field(..., description="Run identifier")
    state: str = Field(..., description="Current state")
    created_at: str = Field(..., description="Creation timestamp")
    metrics_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key metrics"
    )


class RunListResponse(BaseModel):
    """Run list response."""
    runs: List[RunListItem] = Field(..., description="List of runs")
    total: int = Field(..., description="Total number of runs")
    limit: int = Field(..., description="Page size limit")
    offset: int = Field(..., description="Page offset")


class ArtifactListResponse(BaseModel):
    """Artifact list response."""
    artifacts: List[Dict[str, Any]] = Field(..., description="List of artifacts")


class ErrorResponse(BaseModel):
    """Error response."""
    detail: str = Field(..., description="Error detail message")
    status_code: int = Field(..., description="HTTP status code")
    
    class Config:
        schema_extra = {
            "example": {
                "detail": "Run not found",
                "status_code": 404
            }
        }