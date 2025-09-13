"""
FastAPI application for Centrifuge data cleaning pipeline.
Production version with database integration.
"""

import os
import sys
import uuid
import json
import hashlib
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import get_database, close_database, RunManager, ArtifactStore
from core.storage import get_storage_backend, close_storage

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# API Models
class RunSubmitRequest(BaseModel):
    """Request model for run submission."""
    s3_url: Optional[str] = None
    schema_version: str = "v1"
    use_inference: bool = False
    dry_run: bool = False
    llm_columns: List[str] = ["Department", "Account Name"]
    experimental_flags: Dict[str, bool] = {}


class RunSubmitResponse(BaseModel):
    """Response model for run submission."""
    run_id: str
    status: str
    message: str
    queue_position: Optional[int] = None


class RunStatusResponse(BaseModel):
    """Response model for run status."""
    run_id: str
    state: str
    phase_progress: Dict[str, Any]
    metrics: Dict[str, Any]
    error_message: Optional[str]
    error_code: Optional[str]
    created_at: str
    completed_at: Optional[str]


class ArtifactListResponse(BaseModel):
    """Response model for artifact listing."""
    run_id: str
    artifacts: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database: str
    storage: str
    timestamp: str


# Global resources
db_pool = None
storage = None
run_manager = None
artifact_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    Initialize resources on startup, cleanup on shutdown.
    """
    global db_pool, storage, run_manager, artifact_store

    logger.info("Starting Centrifuge API")

    try:
        # initialize database
        db_pool = await get_database()
        run_manager = RunManager(db_pool)
        artifact_store = ArtifactStore(db_pool)

        # initialize storage
        storage = await get_storage_backend()

        logger.info("API resources initialized")

    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise

    yield

    # cleanup on shutdown
    logger.info("Shutting down Centrifuge API")

    if storage:
        await close_storage()

    if db_pool:
        await close_database()

    logger.info("API shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title="Centrifuge API",
    description="Data cleaning pipeline with deterministic rules and LLM assistance",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Checks database and storage connectivity.
    """
    health = {
        "status": "healthy",
        "database": "unknown",
        "storage": "unknown",
        "timestamp": datetime.utcnow().isoformat()
    }

    # check database
    try:
        if db_pool and db_pool.is_healthy:
            if await db_pool.health_check():
                health["database"] = "healthy"
            else:
                health["database"] = "unhealthy"
                health["status"] = "degraded"
        else:
            health["database"] = "unavailable"
            health["status"] = "unhealthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health["database"] = "error"
        health["status"] = "unhealthy"

    # check storage
    try:
        if storage:
            # try to list files in a test prefix
            await storage.list_files("health-check/")
            health["storage"] = "healthy"
        else:
            health["storage"] = "unavailable"
            health["status"] = "unhealthy"
    except Exception as e:
        logger.error(f"Storage health check failed: {e}")
        health["storage"] = "error"
        if health["status"] == "healthy":
            health["status"] = "degraded"

    # return appropriate status code
    if health["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health)

    return health


@app.post("/runs", response_model=RunSubmitResponse)
async def submit_run(
    file: Optional[UploadFile] = File(None),
    s3_url: Optional[str] = Form(None),
    schema_version: str = Form("v1"),
    use_inference: bool = Form(False),
    dry_run: bool = Form(False),
    llm_columns: Optional[str] = Form("Department,Account Name")
):
    """
    Submit a new data cleaning run.

    Accepts either:
    - Multipart file upload
    - S3 URL reference

    The run is queued for asynchronous processing by workers.
    """
    logger.info(f"Received run submission (file={file is not None}, s3_url={s3_url is not None})")

    # validate input
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

    # process LLM columns
    llm_column_list = []
    if llm_columns:
        llm_column_list = [col.strip() for col in llm_columns.split(",")]

    # prepare options
    options = {
        "schema_version": schema_version,
        "use_inference": use_inference,
        "dry_run": dry_run,
        "llm_columns": llm_column_list
    }

    try:
        # handle file upload
        if file:
            # read and validate file
            content = await file.read()

            # check size limit (50MB)
            if len(content) > 50 * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail="File size exceeds 50MB limit"
                )

            # calculate hash
            input_hash = hashlib.sha256(content).hexdigest()

            # upload to storage
            storage_path = f"inputs/{input_hash[:2]}/{input_hash}/input.csv"
            temp_path = f"/tmp/upload_{input_hash}.csv"

            # save temporarily
            with open(temp_path, "wb") as f:
                f.write(content)

            # upload to storage
            await storage.upload(temp_path, storage_path)

            # cleanup temp file
            os.remove(temp_path)

            logger.info(f"Uploaded file to {storage_path}")

        else:
            # handle S3 URL
            if not s3_url.startswith("s3://"):
                raise HTTPException(
                    status_code=400,
                    detail="S3 URL must start with s3://"
                )

            # for S3 URL, use URL hash as input hash
            input_hash = hashlib.sha256(s3_url.encode()).hexdigest()
            options["s3_url"] = s3_url

        # create run in database
        run_id = await run_manager.create_run(
            input_hash=input_hash,
            options=options,
            schema_version=schema_version
        )

        # get queue position (approximate)
        queue_position = await _get_queue_position(run_id)

        logger.info(f"Created run {run_id} with input hash {input_hash}")

        return RunSubmitResponse(
            run_id=run_id,
            status="queued",
            message="Run queued for processing",
            queue_position=queue_position
        )

    except Exception as e:
        logger.error(f"Failed to submit run: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit run: {str(e)}"
        )


@app.get("/runs/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str):
    """
    Get status of a specific run.

    Returns current state, progress, and metrics.
    """
    try:
        # validate UUID
        try:
            uuid.UUID(run_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid run ID format"
            )

        # get run status
        status = await run_manager.get_run_status(run_id)

        if not status:
            raise HTTPException(
                status_code=404,
                detail="Run not found"
            )

        return RunStatusResponse(
            run_id=run_id,
            state=status["state"],
            phase_progress=status.get("phase_progress", {}),
            metrics=status.get("metrics", {}),
            error_message=status.get("error_message"),
            error_code=status.get("error_code"),
            created_at=status["created_at"],
            completed_at=status.get("completed_at")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get run status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get run status: {str(e)}"
        )


@app.get("/runs/{run_id}/artifacts", response_model=ArtifactListResponse)
async def list_artifacts(run_id: str):
    """
    List artifacts for a run.

    Returns metadata for all generated artifacts.
    """
    try:
        # validate UUID
        try:
            uuid.UUID(run_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid run ID format"
            )

        # check run exists
        status = await run_manager.get_run_status(run_id)
        if not status:
            raise HTTPException(
                status_code=404,
                detail="Run not found"
            )

        # get artifacts
        artifacts = await artifact_store.list_artifacts(run_id)

        return ArtifactListResponse(
            run_id=run_id,
            artifacts=artifacts
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list artifacts: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list artifacts: {str(e)}"
        )


@app.get("/runs/{run_id}/artifacts/{artifact_type}/download")
async def download_artifact(
    run_id: str,
    artifact_type: str
):
    """
    Download a specific artifact.

    Streams the artifact file with appropriate content type.
    """
    try:
        # validate UUID
        try:
            uuid.UUID(run_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid run ID format"
            )

        # validate artifact type
        valid_types = ["cleaned", "errors", "audit", "summary", "manifest", "metrics"]
        if artifact_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid artifact type. Must be one of: {valid_types}"
            )

        # get artifacts
        artifacts = await artifact_store.list_artifacts(run_id)

        # find requested artifact
        artifact = None
        for a in artifacts:
            if a["type"] == artifact_type:
                artifact = a
                break

        if not artifact:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact '{artifact_type}' not found for run {run_id}"
            )

        # download from storage
        temp_path = f"/tmp/download_{run_id}_{artifact_type}"
        await storage.download(artifact["storage_path"], temp_path)

        # determine content type
        mime_type = artifact.get("mime_type", "application/octet-stream")

        # stream file
        def iterfile():
            with open(temp_path, "rb") as f:
                yield from f
            # cleanup after streaming
            os.remove(temp_path)

        return StreamingResponse(
            iterfile(),
            media_type=mime_type,
            headers={
                "Content-Disposition": f"attachment; filename={artifact['file_name']}"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download artifact: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download artifact: {str(e)}"
        )


@app.get("/runs")
async def list_runs(
    state: Optional[str] = Query(None, description="Filter by state"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of runs"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    List recent runs with optional filtering.

    Returns a paginated list of runs.
    """
    try:
        # validate state if provided
        valid_states = ["queued", "running", "succeeded", "failed", "partial"]
        if state and state not in valid_states:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid state. Must be one of: {valid_states}"
            )

        # build query
        query = """
            SELECT id::text, status, created_at, completed_at,
                   phase_progress
            FROM runs
        """

        conditions = []
        params = []
        param_count = 0

        if state:
            param_count += 1
            conditions.append(f"status = ${param_count}")
            params.append(state)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC"

        param_count += 1
        query += f" LIMIT ${param_count}"
        params.append(limit)

        param_count += 1
        query += f" OFFSET ${param_count}"
        params.append(offset)

        # execute query
        rows = await db_pool.fetch(query, *params)

        # format results
        runs = []
        for row in rows:
            runs.append({
                "run_id": row["id"],
                "state": row["status"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
                "phase_progress": json.loads(row["phase_progress"]) if row["phase_progress"] else {},
                "metrics": {}  # TODO: join with metrics table if needed
            })

        return {
            "runs": runs,
            "limit": limit,
            "offset": offset,
            "total": len(runs)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list runs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list runs: {str(e)}"
        )


async def _get_queue_position(run_id: str) -> Optional[int]:
    """Get approximate queue position for a run."""
    try:
        query = """
            SELECT COUNT(*) as position
            FROM runs
            WHERE status = 'queued'
              AND created_at < (
                  SELECT created_at FROM runs WHERE id = $1::uuid
              )
        """

        position = await db_pool.fetchval(query, run_id)
        return position + 1 if position is not None else None

    except Exception as e:
        logger.error(f"Failed to get queue position: {e}")
        return None


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
