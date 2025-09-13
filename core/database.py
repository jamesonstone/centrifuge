"""
Database connection module for Centrifuge.
Provides connection pooling, health checks, and query helpers.
"""

import os
import logging
import asyncio
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import asyncpg
from asyncpg import Pool, Connection
import json

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database operations"""
    pass


class DatabaseConnectionError(DatabaseError):
    """Failed to connect to database"""
    pass


class DatabasePool:
    """
    Manages PostgreSQL connection pool with health checks.
    """

    def __init__(self):
        self.pool: Optional[Pool] = None
        self.dsn: Optional[str] = None
        self._healthy: bool = False

    async def initialize(self,
                        host: str = None,
                        port: int = None,
                        database: str = None,
                        user: str = None,
                        password: str = None,
                        min_size: int = 2,
                        max_size: int = 10,
                        timeout: float = 10.0) -> None:
        """
        Initialize the database connection pool.

        Args:
            host: Database host (default from env POSTGRES_HOST)
            port: Database port (default from env POSTGRES_PORT or 5432)
            database: Database name (default from env POSTGRES_DB)
            user: Database user (default from env POSTGRES_USER)
            password: Database password (default from env POSTGRES_PASSWORD)
            min_size: Minimum pool size
            max_size: Maximum pool size
            timeout: Connection timeout in seconds
        """
        # get connection parameters from env or defaults
        host = host or os.getenv('POSTGRES_HOST', 'postgres')
        port = port or int(os.getenv('POSTGRES_PORT', '5432'))
        database = database or os.getenv('POSTGRES_DB', 'centrifuge')
        user = user or os.getenv('POSTGRES_USER', 'centrifuge')
        password = password or os.getenv('POSTGRES_PASSWORD', 'centrifuge')

        self.dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"

        try:
            logger.info(f"Initializing database pool to {host}:{port}/{database}")
            self.pool = await asyncpg.create_pool(
                dsn=self.dsn,
                min_size=min_size,
                max_size=max_size,
                timeout=timeout,
                command_timeout=timeout
            )

            # test the connection
            await self.health_check()
            logger.info("Database pool initialized successfully")

        except Exception as e:
            self._healthy = False
            logger.error(f"Failed to initialize database pool: {e}")
            raise DatabaseConnectionError(f"Failed to connect to database: {e}")

    async def close(self) -> None:
        """Close the database pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            self._healthy = False
            logger.info("Database pool closed")

    async def health_check(self) -> bool:
        """
        Check database health.

        Returns:
            True if healthy, False otherwise
        """
        if not self.pool:
            self._healthy = False
            return False

        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                self._healthy = (result == 1)
                return self._healthy
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self._healthy = False
            return False

    @property
    def is_healthy(self) -> bool:
        """Check if database is healthy."""
        return self._healthy

    @asynccontextmanager
    async def acquire(self):
        """
        Acquire a connection from the pool.

        Yields:
            Connection object
        """
        if not self.pool:
            raise DatabaseConnectionError("Database pool not initialized")

        async with self.pool.acquire() as conn:
            yield conn

    async def execute(self, query: str, *args) -> str:
        """
        Execute a query without returning results.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Status string
        """
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetchval(self, query: str, *args) -> Any:
        """
        Fetch a single value.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Single value
        """
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """
        Fetch a single row.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Single row or None
        """
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """
        Fetch multiple rows.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            List of rows
        """
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)


class RunManager:
    """
    Manages run lifecycle in the database.
    """

    def __init__(self, db: DatabasePool):
        self.db = db

    async def create_run(self,
                        input_hash: str,
                        options: Dict[str, Any]) -> str:
        """
        Create a new run in queued state.

        Args:
            input_hash: SHA256 hash of input file
            options: Run options

        Returns:
            Run ID
        """
        query = """
            INSERT INTO runs (input_file_hash, options, status)
            VALUES ($1, $2, 'queued')
            RETURNING id::text
        """

        run_id = await self.db.fetchval(
            query,
            input_hash,
            json.dumps(options)
        )

        logger.info(f"Created run {run_id} in queued state")
        return run_id

    async def claim_run(self, worker_id: str, visibility_timeout: int = 300) -> Optional[Dict[str, Any]]:
        """
        Atomically claim a queued run for processing.

        Args:
            worker_id: Worker identifier
            visibility_timeout: Seconds before run can be reclaimed

        Returns:
            Run details or None if no runs available
        """
        query = """
            UPDATE runs
            SET status = 'running',
                worker_id = $1,
                started_at = NOW(),
                heartbeat_at = NOW()
            WHERE id = (
                SELECT id
                FROM runs
                WHERE status = 'queued'
                   OR (status = 'running'
                       AND heartbeat_at < NOW() - INTERVAL '%s seconds')
                ORDER BY created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING id::text, input_file_hash, options
        """

        row = await self.db.fetchrow(query % visibility_timeout, worker_id)

        if row:
            return {
                'run_id': row['id'],
                'input_hash': row['input_file_hash'],
                'options': json.loads(row['options'])
            }

        return None

    async def update_heartbeat(self, run_id: str, worker_id: str) -> bool:
        """
        Update run heartbeat.

        Args:
            run_id: Run ID
            worker_id: Worker identifier

        Returns:
            True if updated, False if not found or not owned by worker
        """
        query = """
            UPDATE runs
            SET heartbeat_at = NOW()
            WHERE id = $1::uuid
              AND worker_id = $2
              AND status = 'running'
        """

        result = await self.db.execute(query, run_id, worker_id)
        return "UPDATE 1" in result

    async def update_progress(self,
                            run_id: str,
                            phase: str,
                            percent: int,
                            metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Update run progress.

        Args:
            run_id: Run ID
            phase: Current phase name
            percent: Progress percentage
            metrics: Optional metrics to update
        """
        phase_progress = {'phase': phase, 'percent': percent}

        # For now, ignore metrics parameter since metrics are stored in separate table
        # TODO: Store metrics in metrics table if provided
        query = """
            UPDATE runs
            SET phase_progress = $2,
                updated_at = NOW()
            WHERE id = $1::uuid
        """
        await self.db.execute(query, run_id, json.dumps(phase_progress))

    async def complete_run(self,
                          run_id: str,
                          state: str,
                          metrics: Dict[str, Any],
                          error_message: Optional[str] = None,
                          error_code: Optional[str] = None) -> None:
        """
        Mark run as completed.

        Args:
            run_id: Run ID
            state: Final state (succeeded, failed, partial)
            metrics: Final metrics
            error_message: Optional error message
            error_code: Optional error code
        """
        query = """
            UPDATE runs
            SET status = $2,
                completed_at = NOW(),
                updated_at = NOW()
            WHERE id = $1::uuid
        """

        await self.db.execute(
            query,
            run_id,
            state
        )

        logger.info(f"Run {run_id} completed with state {state}")

    async def get_run_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get run status and progress.

        Args:
            run_id: Run ID

        Returns:
            Run status or None if not found
        """
        query = """
            SELECT id::text, status, phase_progress,
                   created_at, completed_at
            FROM runs
            WHERE id = $1::uuid
        """

        row = await self.db.fetchrow(query, run_id)

        if row:
            return {
                'run_id': row['id'],
                'state': row['status'],
                'phase_progress': json.loads(row['phase_progress']) if row['phase_progress'] else {},
                'metrics': {},  # TODO: fetch from metrics table if needed
                'error_message': None,  # TODO: extract from error_breakdown if needed
                'error_code': None,  # TODO: extract from error_breakdown if needed
                'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                'completed_at': row['completed_at'].isoformat() if row['completed_at'] else None
            }

        return None

    async def check_cancel_requested(self, run_id: str) -> bool:
        """
        Check if cancellation requested for run.

        Args:
            run_id: Run ID

        Returns:
            True if cancel requested
        """
        # for now, we don't have a cancel_requested field, so always return false
        # this is a placeholder for future implementation
        return False


class CanonicalMappingCache:
    """
    Manages canonical mapping cache in database.
    """

    def __init__(self, db: DatabasePool):
        self.db = db

    async def get_mapping(self,
                         column_name: str,
                         variant_value: str,
                         model_id: str = "gpt-4") -> Optional[str]:
        """
        Lookup canonical mapping from cache.

        Args:
            column_name: Column name
            variant_value: Variant value to map
            model_id: LLM model ID

        Returns:
            Canonical value or None if not cached
        """
        query = """
            SELECT canonical_value
            FROM canonical_mappings
            WHERE column_name = $1
              AND variant_value = $2
              AND model_id = $3
              AND superseded_at IS NULL
            ORDER BY created_at DESC
            LIMIT 1
        """

        result = await self.db.fetchval(query, column_name, variant_value, model_id)

        if result:
            # update usage stats
            await self._update_usage(column_name, variant_value, model_id)

        return result

    async def store_mapping(self,
                          column_name: str,
                          variant_value: str,
                          canonical_value: str,
                          model_id: str = "gpt-4",
                          prompt_version: str = "v1",
                          confidence: float = 1.0,
                          source: str = "llm") -> None:
        """
        Store canonical mapping in cache.

        Args:
            column_name: Column name
            variant_value: Variant value
            canonical_value: Canonical value
            model_id: LLM model ID
            prompt_version: Prompt version
            confidence: Confidence score
            source: Source of mapping
        """
        # first, supersede any existing mapping
        supersede_query = """
            UPDATE canonical_mappings
            SET superseded_at = NOW()
            WHERE column_name = $1
              AND variant_value = $2
              AND model_id = $3
              AND superseded_at IS NULL
        """

        await self.db.execute(supersede_query, column_name, variant_value, model_id)

        # insert new mapping
        insert_query = """
            INSERT INTO canonical_mappings
                (column_name, variant_value, canonical_value, model_id,
                 prompt_version, confidence, source)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (column_name, variant_value, model_id, prompt_version)
            DO UPDATE SET
                use_count = canonical_mappings.use_count + 1,
                last_used_at = NOW()
        """

        await self.db.execute(
            insert_query,
            column_name,
            variant_value,
            canonical_value,
            model_id,
            prompt_version,
            confidence,
            source
        )

    async def _update_usage(self, column_name: str, variant_value: str, model_id: str) -> None:
        """Update usage statistics for a mapping."""
        query = """
            UPDATE canonical_mappings
            SET use_count = use_count + 1,
                last_used_at = NOW()
            WHERE column_name = $1
              AND variant_value = $2
              AND model_id = $3
              AND superseded_at IS NULL
        """

        await self.db.execute(query, column_name, variant_value, model_id)

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Cache statistics
        """
        query = """
            SELECT
                COUNT(*) as total_mappings,
                COUNT(DISTINCT column_name) as unique_columns,
                COUNT(DISTINCT variant_value) as unique_variants,
                SUM(use_count) as total_uses,
                AVG(confidence)::float as avg_confidence
            FROM canonical_mappings
            WHERE superseded_at IS NULL
        """

        row = await self.db.fetchrow(query)

        return {
            'total_mappings': row['total_mappings'],
            'unique_columns': row['unique_columns'],
            'unique_variants': row['unique_variants'],
            'total_uses': row['total_uses'],
            'avg_confidence': row['avg_confidence']
        }


class ArtifactStore:
    """
    Manages artifact metadata in database.
    """

    def __init__(self, db: DatabasePool):
        self.db = db

    async def register_artifact(self,
                               run_id: str,
                               artifact_type: str,
                               file_name: str,
                               content_hash: str,
                               storage_path: str,
                               size_bytes: int,
                               mime_type: str = "application/octet-stream",
                               row_count: Optional[int] = None) -> str:
        """
        Register an artifact in the database.

        Args:
            run_id: Run ID
            artifact_type: Type of artifact
            file_name: File name
            content_hash: SHA256 hash
            storage_path: MinIO path
            size_bytes: File size in bytes
            mime_type: MIME type
            row_count: Optional row count

        Returns:
            Artifact ID
        """
        query = """
            INSERT INTO artifacts
                (run_id, artifact_type, file_name, content_hash,
                 storage_path, size_bytes, mime_type, row_count)
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id::text
        """

        artifact_id = await self.db.fetchval(
            query,
            run_id,
            artifact_type,
            file_name,
            content_hash,
            storage_path,
            size_bytes,
            mime_type,
            row_count
        )

        logger.info(f"Registered artifact {artifact_id} for run {run_id}")
        return artifact_id

    async def list_artifacts(self, run_id: str) -> List[Dict[str, Any]]:
        """
        List artifacts for a run.

        Args:
            run_id: Run ID

        Returns:
            List of artifact metadata
        """
        query = """
            SELECT id::text, artifact_type, file_name,
                   content_hash, storage_path, size_bytes,
                   mime_type, row_count, created_at
            FROM artifacts
            WHERE run_id = $1::uuid
            ORDER BY created_at
        """

        rows = await self.db.fetch(query, run_id)

        return [
            {
                'id': row['id'],
                'type': row['artifact_type'],
                'file_name': row['file_name'],
                'content_hash': row['content_hash'],
                'storage_path': row['storage_path'],
                'size_bytes': row['size_bytes'],
                'mime_type': row['mime_type'],
                'row_count': row['row_count'],
                'created_at': row['created_at'].isoformat() if row['created_at'] else None
            }
            for row in rows
        ]

    async def get_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """
        Get artifact metadata.

        Args:
            artifact_id: Artifact ID

        Returns:
            Artifact metadata or None
        """
        query = """
            SELECT id::text, run_id::text, artifact_type, file_name,
                   content_hash, storage_path, size_bytes,
                   mime_type, row_count, created_at
            FROM artifacts
            WHERE id = $1::uuid
        """

        row = await self.db.fetchrow(query, artifact_id)

        if row:
            return {
                'id': row['id'],
                'run_id': row['run_id'],
                'type': row['artifact_type'],
                'file_name': row['file_name'],
                'content_hash': row['content_hash'],
                'storage_path': row['storage_path'],
                'size_bytes': row['size_bytes'],
                'mime_type': row['mime_type'],
                'row_count': row['row_count'],
                'created_at': row['created_at'].isoformat() if row['created_at'] else None
            }

        return None


# global database instance
_db_pool: Optional[DatabasePool] = None


async def get_database() -> DatabasePool:
    """
    Get the global database pool instance.

    Returns:
        Database pool

    Raises:
        DatabaseConnectionError: If not initialized
    """
    global _db_pool

    if not _db_pool:
        _db_pool = DatabasePool()
        await _db_pool.initialize()

    if not _db_pool.is_healthy:
        # try to reconnect
        await _db_pool.initialize()

        if not _db_pool.is_healthy:
            raise DatabaseConnectionError("Database is not healthy")

    return _db_pool


async def close_database() -> None:
    """Close the global database pool."""
    global _db_pool

    if _db_pool:
        await _db_pool.close()
        _db_pool = None
