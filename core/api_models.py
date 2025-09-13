"""API request and response models for Centrifuge."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl
from core.models import RunStatus, ErrorCategory, RunOptions


# =====================================================================
# RUN CREATION
# =====================================================================

class CreateRunRequest(BaseModel):
    """Request to create a new cleaning run."""
    
    # Input source (one of these must be provided)
    s3_url: Optional[HttpUrl] = None
    file_content: Optional[str] = None  # Base64 encoded for API
    
    # Schema selection
    schema_name: str = "accounting_v1"
    schema_version: str = "1.0.0"
    
    # Options
    options: RunOptions = Field(default_factory=RunOptions)
    
    # Metadata
    original_filename: Optional[str] = None
    
    def validate_input(self) -> None:
        """Validate that exactly one input source is provided."""
        if not (self.s3_url or self.file_content):
            raise ValueError("Either s3_url or file_content must be provided")
        if self.s3_url and self.file_content:
            raise ValueError("Only one of s3_url or file_content can be provided")


class CreateRunResponse(BaseModel):
    """Response after creating a run."""
    
    run_id: UUID
    run_seq: int
    status: RunStatus
    message: str = "Run created successfully"
    estimated_duration_seconds: Optional[int] = None
    
    # Links
    status_url: str
    artifacts_url: str


# =====================================================================
# RUN STATUS
# =====================================================================

class PhaseProgress(BaseModel):
    """Progress tracking for current phase."""
    
    phase: str
    percent: int
    message: Optional[str] = None


class RunStatusResponse(BaseModel):
    """Response for run status query."""
    
    run_id: UUID
    run_seq: int
    status: RunStatus
    
    # Progress
    phase_progress: PhaseProgress
    
    # Timing
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Input info
    input_file_name: Optional[str] = None
    input_row_count: Optional[int] = None
    
    # Results (if completed)
    total_rows: Optional[int] = None
    cleaned_rows: Optional[int] = None
    quarantined_rows: Optional[int] = None
    success_rate: Optional[float] = None
    
    # Error info (if failed)
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    # Breakdown
    error_breakdown: Optional[Dict[ErrorCategory, int]] = None
    
    # Worker info
    worker_id: Optional[str] = None
    heartbeat_at: Optional[datetime] = None


# =====================================================================
# ARTIFACTS
# =====================================================================

class ArtifactInfo(BaseModel):
    """Information about a single artifact."""
    
    artifact_type: str
    file_name: str
    content_hash: str
    size_bytes: int
    mime_type: str
    download_url: str
    created_at: datetime
    
    # Optional metadata
    row_count: Optional[int] = None


class RunArtifactsResponse(BaseModel):
    """Response listing all artifacts for a run."""
    
    run_id: UUID
    run_seq: int
    status: RunStatus
    artifacts: List[ArtifactInfo]
    
    # Quick access URLs
    cleaned_csv_url: Optional[str] = None
    errors_csv_url: Optional[str] = None
    diff_csv_url: Optional[str] = None
    audit_ndjson_url: Optional[str] = None
    manifest_json_url: Optional[str] = None
    metrics_json_url: Optional[str] = None
    summary_md_url: Optional[str] = None


# =====================================================================
# BATCH OPERATIONS
# =====================================================================

class BatchRunRequest(BaseModel):
    """Request to process multiple files in batch."""
    
    runs: List[CreateRunRequest]
    parallel: bool = False  # Process in parallel or sequential
    stop_on_error: bool = True
    
    # Batch options
    batch_name: Optional[str] = None
    notification_webhook: Optional[HttpUrl] = None


class BatchRunResponse(BaseModel):
    """Response for batch run creation."""
    
    batch_id: UUID
    total_runs: int
    runs_created: List[CreateRunResponse]
    runs_failed: List[Dict[str, Any]]
    message: str


# =====================================================================
# SCHEMA MANAGEMENT
# =====================================================================

class ListSchemasResponse(BaseModel):
    """Response listing available schemas."""
    
    schemas: List[Dict[str, Any]]
    total: int


class GetSchemaResponse(BaseModel):
    """Response with schema details."""
    
    name: str
    version: str
    columns: Dict[str, Any]
    constraints: Dict[str, Any]
    canonical_departments: List[str]
    canonical_accounts: List[str]
    is_active: bool
    created_at: datetime


# =====================================================================
# HEALTH & SYSTEM
# =====================================================================

class SystemStatsResponse(BaseModel):
    """System statistics and health."""
    
    total_runs: int
    runs_queued: int
    runs_running: int
    runs_completed: int
    runs_failed: int
    
    # Worker stats
    active_workers: int
    worker_details: List[Dict[str, Any]]
    
    # Performance
    avg_duration_seconds: float
    success_rate: float
    
    # Storage
    total_artifacts: int
    storage_used_gb: float


# =====================================================================
# ERROR RESPONSES
# =====================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str
    detail: Optional[str] = None
    status_code: int
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None
    
    # Validation errors
    validation_errors: Optional[List[Dict[str, Any]]] = None


class ValidationErrorDetail(BaseModel):
    """Detail about a validation error."""
    
    field: str
    message: str
    value: Any
    constraint: Optional[str] = None