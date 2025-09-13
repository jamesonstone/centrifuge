"""Database models and utilities for Centrifuge."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from uuid_extensions import uuid7

from core.models import RunStatus, SourceType, ErrorCategory
from core.config import settings


class DatabaseManager:
    """Manages database connections and operations."""

    def __init__(self, connection_string: str = None):
        """Initialize database manager."""
        self.connection_string = connection_string or settings.database_url
        self._pool = None

    async def initialize(self):
        """Initialize connection pool."""
        if not self._pool:
            self._pool = await psycopg.AsyncConnectionPool.open(
                self.connection_string,
                min_size=2,
                max_size=10,
            )

    async def close(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def get_connection(self):
        """Get a connection from the pool."""
        if not self._pool:
            await self.initialize()
        return self._pool.connection()

    # =====================================================================
    # RUN OPERATIONS
    # =====================================================================

    async def create_run(
        self,
        input_file_name: str,
        input_file_hash: str,
        input_row_count: int,
        options: Dict[str, Any],
        use_inferred: bool = False,
        dry_run: bool = False,
        llm_columns: List[str] = None
    ) -> Dict[str, Any]:
        """Create a new run in the database."""
        async with await self.get_connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                query = """
                    INSERT INTO runs (
                        input_file_name, input_file_hash, input_row_count,
                        options, use_inferred, dry_run, llm_columns
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s
                    ) RETURNING id, run_seq, status, created_at
                """
                await cur.execute(
                    query,
                    (
                        input_file_name,
                        input_file_hash,
                        input_row_count,
                        json.dumps(options),
                        use_inferred,
                        dry_run,
                        llm_columns or ["Department", "Account Name"]
                    )
                )
                return await cur.fetchone()

    async def get_run(self, run_id: UUID) -> Optional[Dict[str, Any]]:
        """Get run details by ID."""
        async with await self.get_connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                query = """
                    SELECT * FROM runs WHERE id = %s
                """
                await cur.execute(query, (str(run_id),))
                return await cur.fetchone()

    async def claim_run(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Claim a queued run for processing."""
        async with await self.get_connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                query = """
                    SELECT * FROM claim_run(%s)
                """
                await cur.execute(query, (worker_id,))
                result = await cur.fetchone()
                if result:
                    # Convert to full run record
                    run_query = "SELECT * FROM runs WHERE id = %s"
                    await cur.execute(run_query, (result['run_id'],))
                    return await cur.fetchone()
                return None

    async def update_run_status(
        self,
        run_id: UUID,
        status: RunStatus,
        phase_progress: Dict[str, Any] = None,
        **kwargs
    ) -> bool:
        """Update run status and optional fields."""
        async with await self.get_connection() as conn:
            async with conn.cursor() as cur:
                # Build dynamic update query
                fields = ["status = %s", "updated_at = NOW()"]
                values = [status.value]

                if phase_progress:
                    fields.append("phase_progress = %s")
                    values.append(json.dumps(phase_progress))

                # Add any additional fields
                for key, value in kwargs.items():
                    if key in [
                        "total_rows", "cleaned_rows", "quarantined_rows",
                        "rules_fixed_count", "llm_fixed_count", "error_breakdown",
                        "completed_at"
                    ]:
                        fields.append(f"{key} = %s")
                        values.append(value)

                values.append(str(run_id))

                query = f"""
                    UPDATE runs SET {', '.join(fields)}
                    WHERE id = %s
                """
                await cur.execute(query, values)
                return cur.rowcount > 0

    async def update_heartbeat(self, run_id: UUID, worker_id: str) -> bool:
        """Update worker heartbeat for a run."""
        async with await self.get_connection() as conn:
            async with conn.cursor() as cur:
                query = """
                    SELECT * FROM update_heartbeat(%s, %s)
                """
                await cur.execute(query, (str(run_id), worker_id))
                result = await cur.fetchone()
                return result[0] if result else False

    # =====================================================================
    # SCHEMA OPERATIONS
    # =====================================================================

    async def get_schema(self, name: str, version: str) -> Optional[Dict[str, Any]]:
        """Get schema by name and version."""
        async with await self.get_connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                query = """
                    SELECT * FROM schemas
                    WHERE name = %s AND version = %s AND is_active = true
                """
                await cur.execute(query, (name, version))
                return await cur.fetchone()

    async def list_schemas(self) -> List[Dict[str, Any]]:
        """List all active schemas."""
        async with await self.get_connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                query = """
                    SELECT name, version, created_at, updated_at
                    FROM schemas WHERE is_active = true
                    ORDER BY name, version DESC
                """
                await cur.execute(query)
                return await cur.fetchall()

    # =====================================================================
    # CANONICAL MAPPING OPERATIONS
    # =====================================================================

    async def get_canonical_mapping(
        self,
        column_name: str,
        variant_value: str
    ) -> Optional[Dict[str, Any]]:
        """Get approved canonical mapping for a value."""
        async with await self.get_connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                query = """
                    SELECT * FROM canonical_mappings
                    WHERE column_name = %s
                      AND variant_value = %s
                      AND is_approved = true
                      AND superseded_at IS NULL
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                await cur.execute(query, (column_name, variant_value))
                return await cur.fetchone()

    async def create_canonical_mapping(
        self,
        column_name: str,
        variant_value: str,
        canonical_value: str,
        model_id: str,
        prompt_version: str,
        source: SourceType,
        confidence: float = None
    ) -> UUID:
        """Create or update a canonical mapping."""
        async with await self.get_connection() as conn:
            async with conn.cursor() as cur:
                query = """
                    SELECT * FROM get_or_create_mapping(
                        %s, %s, %s, %s, %s, %s, %s
                    )
                """
                await cur.execute(
                    query,
                    (
                        column_name,
                        variant_value,
                        canonical_value,
                        model_id,
                        prompt_version,
                        source.value,
                        confidence
                    )
                )
                result = await cur.fetchone()
                return UUID(result[0]) if result else None

    async def get_mappings_batch(
        self,
        column_name: str,
        variant_values: List[str]
    ) -> Dict[str, str]:
        """Get canonical mappings for multiple values."""
        async with await self.get_connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                query = """
                    SELECT variant_value, canonical_value
                    FROM canonical_mappings
                    WHERE column_name = %s
                      AND variant_value = ANY(%s)
                      AND is_approved = true
                      AND superseded_at IS NULL
                """
                await cur.execute(query, (column_name, variant_values))
                results = await cur.fetchall()
                return {r['variant_value']: r['canonical_value'] for r in results}

    # =====================================================================
    # ARTIFACT OPERATIONS
    # =====================================================================

    async def create_artifact(
        self,
        run_id: UUID,
        artifact_type: str,
        file_name: str,
        content_hash: str,
        storage_path: str,
        size_bytes: int,
        mime_type: str = None,
        row_count: int = None
    ) -> UUID:
        """Create artifact record."""
        async with await self.get_connection() as conn:
            async with conn.cursor() as cur:
                query = """
                    INSERT INTO artifacts (
                        run_id, artifact_type, file_name, content_hash,
                        storage_path, size_bytes, mime_type, row_count
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s
                    ) RETURNING id
                """
                await cur.execute(
                    query,
                    (
                        str(run_id),
                        artifact_type,
                        file_name,
                        content_hash,
                        storage_path,
                        size_bytes,
                        mime_type,
                        row_count
                    )
                )
                result = await cur.fetchone()
                return UUID(result[0]) if result else None

    async def get_run_artifacts(self, run_id: UUID) -> List[Dict[str, Any]]:
        """Get all artifacts for a run."""
        async with await self.get_connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                query = """
                    SELECT * FROM artifacts
                    WHERE run_id = %s
                    ORDER BY created_at
                """
                await cur.execute(query, (str(run_id),))
                return await cur.fetchall()

    # =====================================================================
    # AUDIT OPERATIONS
    # =====================================================================

    async def create_audit_event(
        self,
        run_id: UUID,
        row_uuid: UUID,
        column_name: str,
        before_value: str,
        after_value: str,
        source: SourceType,
        rule_id: str = None,
        contract_id: str = None,
        reason: str = None,
        confidence: float = None
    ) -> UUID:
        """Create audit log entry."""
        async with await self.get_connection() as conn:
            async with conn.cursor() as cur:
                query = """
                    INSERT INTO audit_log (
                        run_id, row_uuid, column_name, before_value, after_value,
                        source, rule_id, contract_id, reason, confidence
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) RETURNING id
                """
                await cur.execute(
                    query,
                    (
                        str(run_id),
                        str(row_uuid),
                        column_name,
                        before_value,
                        after_value,
                        source.value,
                        rule_id,
                        contract_id,
                        reason,
                        confidence
                    )
                )
                result = await cur.fetchone()
                return UUID(result[0]) if result else None

    async def create_audit_events_batch(
        self,
        events: List[Dict[str, Any]]
    ) -> int:
        """Create multiple audit events efficiently."""
        if not events:
            return 0

        async with await self.get_connection() as conn:
            async with conn.cursor() as cur:
                # Use COPY for efficient bulk insert
                columns = [
                    "run_id", "row_uuid", "column_name", "before_value",
                    "after_value", "source", "rule_id", "contract_id",
                    "reason", "confidence"
                ]

                async with cur.copy(
                    f"COPY audit_log ({', '.join(columns)}) FROM STDIN"
                ) as copy:
                    for event in events:
                        await copy.write_row([
                            str(event.get('run_id')),
                            str(event.get('row_uuid')),
                            event.get('column_name'),
                            event.get('before_value'),
                            event.get('after_value'),
                            event.get('source'),
                            event.get('rule_id'),
                            event.get('contract_id'),
                            event.get('reason'),
                            event.get('confidence')
                        ])

                return len(events)

    # =====================================================================
    # METRICS OPERATIONS
    # =====================================================================

    async def create_metrics(
        self,
        run_id: UUID,
        metrics_data: Dict[str, Any]
    ) -> UUID:
        """Create metrics record for a run."""
        async with await self.get_connection() as conn:
            async with conn.cursor() as cur:
                # Build insert query with available fields
                fields = ["run_id"]
                values = [str(run_id)]

                # Map metrics data to database columns
                field_mapping = {
                    "total_cells": "total_cells",
                    "cells_validated": "cells_validated",
                    "cells_modified": "cells_modified",
                    "cells_quarantined": "cells_quarantined",
                    "rules_duration_ms": "rules_duration_ms",
                    "llm_duration_ms": "llm_duration_ms",
                    "total_duration_ms": "total_duration_ms",
                    "llm_calls_count": "llm_calls_count",
                    "llm_tokens_used": "llm_tokens_used",
                    "llm_cost_estimate": "llm_cost_estimate",
                    "cache_hits": "cache_hits",
                    "cache_misses": "cache_misses",
                    "validation_failures": "validation_failures",
                    "llm_contract_failures": "llm_contract_failures",
                    "low_confidence_count": "low_confidence_count",
                    "edit_cap_exceeded_count": "edit_cap_exceeded_count",
                    "parse_errors": "parse_errors"
                }

                for key, db_field in field_mapping.items():
                    if key in metrics_data:
                        fields.append(db_field)
                        values.append(metrics_data[key])

                placeholders = ", ".join(["%s"] * len(values))
                fields_str = ", ".join(fields)

                query = f"""
                    INSERT INTO metrics ({fields_str})
                    VALUES ({placeholders})
                    RETURNING id
                """
                await cur.execute(query, values)
                result = await cur.fetchone()
                return UUID(result[0]) if result else None

    # =====================================================================
    # STATISTICS
    # =====================================================================

    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics."""
        async with await self.get_connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                query = """
                    SELECT
                        COUNT(*) as total_runs,
                        COUNT(*) FILTER (WHERE status = 'queued') as runs_queued,
                        COUNT(*) FILTER (WHERE status = 'running') as runs_running,
                        COUNT(*) FILTER (WHERE status = 'succeeded') as runs_succeeded,
                        COUNT(*) FILTER (WHERE status = 'partial') as runs_partial,
                        COUNT(*) FILTER (WHERE status = 'failed') as runs_failed,
                        AVG(EXTRACT(EPOCH FROM (completed_at - started_at)))
                            FILTER (WHERE completed_at IS NOT NULL) as avg_duration_seconds,
                        COUNT(DISTINCT worker_id) FILTER (WHERE status = 'running') as active_workers
                    FROM runs
                """
                await cur.execute(query)
                return await cur.fetchone()


# Global database manager instance
db = DatabaseManager()
