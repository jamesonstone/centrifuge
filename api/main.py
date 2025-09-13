"""FastAPI application for Centrifuge data cleaning pipeline."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import structlog

from core.models import RunState, RunManifest
from core.config import settings
from api.models import (
    RunSubmitRequest, RunSubmitResponse,
    RunStatusResponse, RunArtifact,
    HealthResponse
)

logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="Centrifuge API",
    description="Data cleaning pipeline with deterministic rules and LLM assistance",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


class RunManager:
    """Manages run state in memory (for PoC - would use DB in production)."""
    
    def __init__(self):
        self.runs: Dict[UUID, Dict[str, Any]] = {}
        
    def create_run(
        self,
        input_hash: str,
        schema_version: str,
        options: Dict[str, Any]
    ) -> UUID:
        """Create a new run."""
        run_id = uuid4()
        
        self.runs[run_id] = {
            'run_id': run_id,
            'created_at': datetime.now(),
            'state': RunState.QUEUED,
            'input_hash': input_hash,
            'schema_version': schema_version,
            'options': options,
            'phase_progress': {},
            'metrics': {},
            'artifacts': {},
            'error': None
        }
        
        return run_id
    
    def get_run(self, run_id: UUID) -> Optional[Dict[str, Any]]:
        """Get run by ID."""
        return self.runs.get(run_id)
    
    def update_run_state(
        self,
        run_id: UUID,
        state: RunState,
        phase_progress: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Update run state."""
        if run_id in self.runs:
            self.runs[run_id]['state'] = state
            if phase_progress:
                self.runs[run_id]['phase_progress'] = phase_progress
            if metrics:
                self.runs[run_id]['metrics'] = metrics
            if artifacts:
                self.runs[run_id]['artifacts'] = artifacts
            if error:
                self.runs[run_id]['error'] = error


# Initialize run manager
run_manager = RunManager()


@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns 200 with service status.
    """
    # In production, would check DB connection
    # For now, just return healthy
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/runs", response_model=RunSubmitResponse)
async def submit_run(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    s3_url: Optional[str] = Form(None),
    schema_version: str = Form("1.0.0"),
    use_inferred: bool = Form(False),
    dry_run: bool = Form(False),
    llm_columns: Optional[str] = Form(None)
):
    """
    Submit a new data cleaning run.
    
    Accepts either:
    - Multipart file upload
    - S3 URL reference
    
    Options:
    - use_inferred: Enable schema inference (experimental)
    - dry_run: Validate without applying changes
    - llm_columns: Comma-separated list of columns for LLM processing
    """
    logger.info("Received run submission",
               has_file=file is not None,
               has_s3_url=s3_url is not None)
    
    # Validate input
    if not file and not s3_url:
        raise HTTPException(
            status_code=400,
            detail="Either file upload or s3_url is required"
        )
    
    if file and s3_url:
        raise HTTPException(
            status_code=400,
            detail="Provide either file upload or s3_url, not both"
        )
    
    # Process file upload
    input_hash = ""
    if file:
        # Read file content
        content = await file.read()
        
        # Validate file size
        if len(content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum of {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Calculate content hash
        input_hash = hashlib.sha256(content).hexdigest()
        
        # Save to temporary location (in production, would stream to S3)
        temp_path = Path(f"temp/{input_hash}.csv")
        temp_path.parent.mkdir(exist_ok=True)
        temp_path.write_bytes(content)
        
        logger.info("File uploaded",
                   filename=file.filename,
                   size=len(content),
                   hash=input_hash)
    
    elif s3_url:
        # Validate S3 URL format
        if not s3_url.startswith("s3://"):
            raise HTTPException(
                status_code=400,
                detail="S3 URL must start with s3://"
            )
        
        # In production, would download and hash S3 content
        # For PoC, use URL as hash
        input_hash = hashlib.sha256(s3_url.encode()).hexdigest()
        
        logger.info("S3 URL provided",
                   url=s3_url,
                   hash=input_hash)
    
    # Parse options
    options = {
        'use_inferred': use_inferred,
        'dry_run': dry_run,
        'llm_columns': llm_columns.split(',') if llm_columns else ['department', 'account_name']
    }
    
    # Create run
    run_id = run_manager.create_run(
        input_hash=input_hash,
        schema_version=schema_version,
        options=options
    )
    
    # Queue background processing
    background_tasks.add_task(
        process_run_async,
        run_id,
        input_hash,
        schema_version,
        options
    )
    
    logger.info("Run created",
               run_id=str(run_id),
               state=RunState.QUEUED.value)
    
    return RunSubmitResponse(
        run_id=run_id,
        state=RunState.QUEUED,
        message="Run queued for processing",
        estimated_duration_seconds=30
    )


async def process_run_async(
    run_id: UUID,
    input_hash: str,
    schema_version: str,
    options: Dict[str, Any]
):
    """
    Process run asynchronously (mock implementation).
    
    In production, this would be handled by a worker pool.
    """
    try:
        # Update state to running
        run_manager.update_run_state(run_id, RunState.RUNNING)
        
        # Simulate processing phases
        import asyncio
        phases = [
            "ingest", "validation_1", "rules_engine",
            "residual_planner", "llm_adapter", "apply_patches",
            "validation_2", "artifacts"
        ]
        
        phase_progress = {}
        for i, phase in enumerate(phases):
            await asyncio.sleep(0.5)  # Simulate work
            phase_progress[phase] = 100
            run_manager.update_run_state(
                run_id,
                RunState.RUNNING,
                phase_progress=phase_progress
            )
        
        # Generate mock artifacts
        artifacts = {
            'cleaned.csv': {
                'name': 'cleaned.csv',
                'size_bytes': 1024,
                'content_hash': 'abc123',
                'storage_path': f'artifacts/{run_id}/cleaned.csv'
            },
            'errors.csv': {
                'name': 'errors.csv',
                'size_bytes': 512,
                'content_hash': 'def456',
                'storage_path': f'artifacts/{run_id}/errors.csv'
            },
            'diff.csv': {
                'name': 'diff.csv',
                'size_bytes': 2048,
                'content_hash': 'ghi789',
                'storage_path': f'artifacts/{run_id}/diff.csv'
            },
            'audit.ndjson': {
                'name': 'audit.ndjson',
                'size_bytes': 4096,
                'content_hash': 'jkl012',
                'storage_path': f'artifacts/{run_id}/audit.ndjson'
            },
            'manifest.json': {
                'name': 'manifest.json',
                'size_bytes': 256,
                'content_hash': 'mno345',
                'storage_path': f'artifacts/{run_id}/manifest.json'
            },
            'metrics.json': {
                'name': 'metrics.json',
                'size_bytes': 512,
                'content_hash': 'pqr678',
                'storage_path': f'artifacts/{run_id}/metrics.json'
            },
            'summary.md': {
                'name': 'summary.md',
                'size_bytes': 1024,
                'content_hash': 'stu901',
                'storage_path': f'artifacts/{run_id}/summary.md'
            }
        }
        
        # Generate mock metrics
        metrics = {
            'total_rows': 1000,
            'clean_rows': 950,
            'quarantined_rows': 50,
            'success_rate': 0.95,
            'rules_fixed_count': 800,
            'llm_fixed_count': 150,
            'cache_hit_rate': 0.7,
            'total_duration_seconds': len(phases) * 0.5
        }
        
        # Update to succeeded
        run_manager.update_run_state(
            run_id,
            RunState.SUCCEEDED,
            phase_progress=phase_progress,
            metrics=metrics,
            artifacts=artifacts
        )
        
        logger.info("Run completed successfully",
                   run_id=str(run_id),
                   duration=metrics['total_duration_seconds'])
        
    except Exception as e:
        logger.error("Run failed",
                    run_id=str(run_id),
                    error=str(e))
        
        run_manager.update_run_state(
            run_id,
            RunState.FAILED,
            error=str(e)
        )


@app.get("/runs/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: UUID):
    """
    Get status and details for a specific run.
    
    Returns:
    - Current state (queued, running, succeeded, partial, failed)
    - Progress by phase
    - Metrics summary
    - Artifact links
    - Error details if failed
    """
    run = run_manager.get_run(run_id)
    
    if not run:
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id} not found"
        )
    
    # Build response
    response = RunStatusResponse(
        run_id=run['run_id'],
        state=run['state'],
        created_at=run['created_at'].isoformat(),
        phase_progress=run.get('phase_progress', {}),
        metrics=run.get('metrics', {}),
        artifacts=[
            RunArtifact(
                name=artifact['name'],
                size_bytes=artifact['size_bytes'],
                content_hash=artifact['content_hash'],
                download_url=f"/runs/{run_id}/artifacts/{artifact['name']}"
            )
            for artifact in run.get('artifacts', {}).values()
        ],
        error=run.get('error')
    )
    
    return response


@app.get("/runs/{run_id}/artifacts")
async def list_artifacts(run_id: UUID):
    """
    List all artifacts for a run.
    
    Returns array of artifact metadata with download links.
    """
    run = run_manager.get_run(run_id)
    
    if not run:
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id} not found"
        )
    
    if run['state'] not in [RunState.SUCCEEDED, RunState.PARTIAL]:
        raise HTTPException(
            status_code=400,
            detail=f"Run is in state {run['state'].value}, artifacts not available"
        )
    
    artifacts = []
    for artifact in run.get('artifacts', {}).values():
        artifacts.append({
            'name': artifact['name'],
            'size_bytes': artifact['size_bytes'],
            'content_hash': artifact['content_hash'],
            'download_url': f"/runs/{run_id}/artifacts/{artifact['name']}"
        })
    
    return {'artifacts': artifacts}


@app.get("/runs/{run_id}/artifacts/{artifact_name}")
async def download_artifact(run_id: UUID, artifact_name: str):
    """
    Download a specific artifact.
    
    Streams the artifact content with appropriate content-type.
    """
    run = run_manager.get_run(run_id)
    
    if not run:
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id} not found"
        )
    
    artifacts = run.get('artifacts', {})
    if artifact_name not in artifacts:
        raise HTTPException(
            status_code=404,
            detail=f"Artifact {artifact_name} not found for run {run_id}"
        )
    
    # In production, would stream from MinIO/S3
    # For PoC, create mock content
    artifact = artifacts[artifact_name]
    
    # Determine content type
    content_type = "text/plain"
    if artifact_name.endswith('.csv'):
        content_type = "text/csv"
    elif artifact_name.endswith('.json'):
        content_type = "application/json"
    elif artifact_name.endswith('.ndjson'):
        content_type = "application/x-ndjson"
    elif artifact_name.endswith('.md'):
        content_type = "text/markdown"
    
    # Generate mock content based on artifact type
    if artifact_name == 'cleaned.csv':
        content = "transaction_id,department,amount\nTXN001,Sales,100.00\nTXN002,Finance,200.00"
    elif artifact_name == 'errors.csv':
        content = "row_number,error_category,error_summary\n11,validation_failure,Invalid department"
    elif artifact_name == 'diff.csv':
        content = "row_number,column,before_value,after_value,source,reason\n1,department,sales,Sales,rule,Normalized case"
    elif artifact_name == 'audit.ndjson':
        content = '{"event_id":"123","event_type":"transformation","column":"department","before":"sales","after":"Sales"}'
    elif artifact_name == 'manifest.json':
        content = json.dumps({
            'run_id': str(run_id),
            'schema_version': '1.0.0',
            'model_version': 'gpt-5',
            'seed': 42
        }, indent=2)
    elif artifact_name == 'metrics.json':
        content = json.dumps(run.get('metrics', {}), indent=2)
    elif artifact_name == 'summary.md':
        content = f"""# Centrifuge Run Summary
        
Run ID: {run_id}
Date: {datetime.now().isoformat()}

## Overview
- Total Rows: {run.get('metrics', {}).get('total_rows', 0)}
- Clean Rows: {run.get('metrics', {}).get('clean_rows', 0)}
- Quarantined Rows: {run.get('metrics', {}).get('quarantined_rows', 0)}
"""
    else:
        content = f"Mock content for {artifact_name}"
    
    # Return streaming response
    return StreamingResponse(
        iter([content.encode()]),
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename={artifact_name}",
            "Content-Length": str(len(content))
        }
    )


@app.get("/runs")
async def list_runs(
    limit: int = 10,
    offset: int = 0,
    state: Optional[RunState] = None
):
    """
    List runs with optional filtering.
    
    Parameters:
    - limit: Maximum number of runs to return
    - offset: Number of runs to skip
    - state: Filter by run state
    """
    # Get all runs
    all_runs = list(run_manager.runs.values())
    
    # Filter by state if provided
    if state:
        all_runs = [r for r in all_runs if r['state'] == state]
    
    # Sort by created_at descending
    all_runs.sort(key=lambda r: r['created_at'], reverse=True)
    
    # Apply pagination
    paginated_runs = all_runs[offset:offset + limit]
    
    # Build response
    runs = []
    for run in paginated_runs:
        runs.append({
            'run_id': str(run['run_id']),
            'state': run['state'].value,
            'created_at': run['created_at'].isoformat(),
            'metrics_summary': {
                'total_rows': run.get('metrics', {}).get('total_rows', 0),
                'success_rate': run.get('metrics', {}).get('success_rate', 0)
            }
        })
    
    return {
        'runs': runs,
        'total': len(all_runs),
        'limit': limit,
        'offset': offset
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)