"""Centrifuge API - CSV cleaning with deterministic rules and contracted LLM assists."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict

import psycopg
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from core.config import settings

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Database connection pool
db_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global db_pool
    
    # Startup
    logger.info("Starting Centrifuge API", 
                environment=settings.environment,
                worker_id=settings.worker_id)
    
    # Initialize database connection pool
    try:
        db_pool = await psycopg.AsyncConnectionPool.open(
            settings.database_url,
            min_size=2,
            max_size=10,
        )
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.error("Failed to initialize database pool", error=str(e))
        if settings.is_production:
            raise
    
    # Note: MinIO and LiteLLM connections are established on-demand
    # per the PoC requirements to keep health checks simple
    
    yield
    
    # Shutdown
    logger.info("Shutting down Centrifuge API")
    if db_pool:
        await db_pool.close()
        logger.info("Database connection pool closed")


# Create FastAPI app
app = FastAPI(
    title="Centrifuge",
    description="Trustworthy CSV cleaning with deterministic rules and contracted LLM assists",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Authentication middleware hook (pass-through for PoC)
@app.middleware("http")
async def auth_middleware(request, call_next):
    """
    Authentication middleware hook.
    
    Currently pass-through for PoC.
    In production, implement JWT/API key validation here.
    """
    # TODO: Implement authentication
    # Example structure for future implementation:
    # auth_header = request.headers.get("Authorization")
    # if auth_header:
    #     try:
    #         token = auth_header.split(" ")[1]
    #         user = await validate_jwt(token)
    #         request.state.user = user
    #     except Exception:
    #         return JSONResponse(
    #             status_code=401,
    #             content={"detail": "Invalid authentication credentials"}
    #         )
    
    response = await call_next(request)
    return response


@app.get("/healthz", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns 200 if the service is up and can perform a basic database ping.
    Per PoC requirements, we skip MinIO and LiteLLM checks to keep it simple.
    Failures in those services will surface naturally during run execution.
    """
    health_status = {
        "status": "healthy",
        "environment": settings.environment,
        "worker_id": settings.worker_id,
        "checks": {}
    }
    
    # Database health check
    try:
        async with db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                result = await cur.fetchone()
                if result and result[0] == 1:
                    health_status["checks"]["database"] = "healthy"
                else:
                    health_status["checks"]["database"] = "unhealthy"
                    health_status["status"] = "degraded"
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        health_status["checks"]["database"] = "unhealthy"
        health_status["status"] = "unhealthy"
        
        # Return 503 if database is down
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection failed"
        )
    
    # Note: In production, we would add more checks here:
    # - MinIO connectivity
    # - LiteLLM availability
    # - Redis cache status
    # - Queue depth monitoring
    # But per requirements, we keep it simple for the PoC
    
    return health_status


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Centrifuge API",
        "version": "0.1.0",
        "description": "CSV cleaning with deterministic rules and contracted LLM assists",
        "endpoints": {
            "health": "/healthz",
            "docs": "/docs",
            "runs": "/runs",
        }
    }


# Placeholder for run endpoints (to be implemented in Phase 10)
@app.post("/runs", tags=["Runs"], status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def create_run():
    """Create a new cleaning run. (To be implemented)"""
    return {"message": "Endpoint will be implemented in Phase 10"}


@app.get("/runs/{run_id}", tags=["Runs"], status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def get_run(run_id: str):
    """Get run status and details. (To be implemented)"""
    return {"message": "Endpoint will be implemented in Phase 10"}


@app.get("/runs/{run_id}/artifacts", tags=["Runs"], status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def get_run_artifacts(run_id: str):
    """Get run artifacts. (To be implemented)"""
    return {"message": "Endpoint will be implemented in Phase 10"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=settings.is_development,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
        }
    )