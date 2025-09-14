"""
Data processing pipeline for Centrifuge.
Orchestrates the flow through all processing phases.
"""

import os
import io
import logging
import hashlib
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

from core.ingest import DataIngestor
from core.rules import RuleEngine
from core.planner import ResidualPlanner
from core.llm_client import get_llm_adapter
from core.quarantine import QuarantineManager
from core.artifacts import ArtifactManager
from core.database import CanonicalMappingCache, ArtifactStore
from core.storage import StorageBackend
from core.models import Schema, ColumnDefinition, ColumnType, ColumnPolicy

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from pipeline processing."""
    total_rows: int = 0
    cleaned_rows: int = 0
    quarantined_rows: int = 0
    rules_fixed_count: int = 0
    llm_fixed_count: int = 0
    error_breakdown: Dict[str, int] = field(default_factory=dict)
    data: Any = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    quarantine_df: Any = None  # DataFrame of quarantined rows
    input_file_path: Optional[str] = None  # Path to original input file


class DataPipeline:
    """
    Orchestrates the data cleaning pipeline.
    """

    def __init__(self,
                 storage: StorageBackend,
                 cache: Optional[CanonicalMappingCache] = None,
                 artifact_store: Optional[ArtifactStore] = None):
        """
        Initialize pipeline.

        Args:
            storage: Storage backend for artifacts
            cache: Optional canonical mapping cache
            artifact_store: Optional artifact metadata store
        """
        self.storage = storage
        self.cache = cache
        self.artifact_store = artifact_store
        self.llm_adapter = None

    async def ingest(self,
                    file_path: str,
                    schema_version: str = "v1",
                    use_inference: bool = False) -> PipelineResult:
        """
        Phase 1: Ingest and validate input data.

        Args:
            file_path: Path to input CSV file
            schema_version: Schema version to use
            use_inference: Whether to use schema inference

        Returns:
            Pipeline result with ingested data
        """
        logger.info(f"Phase 1: Ingesting {file_path}")

        ingestor = DataIngestor()

        # check file size (50k row limit)
        row_count = 0
        with open(file_path, 'r') as f:
            for _ in f:
                row_count += 1
                if row_count > 50000:
                    raise ValueError("Input file exceeds 50,000 row limit")

        # ingest data
        if use_inference:
            logger.info("Using schema inference")
            ingest_result = await asyncio.to_thread(
                ingestor.ingest_with_inference,
                file_path
            )
        else:
            ingest_result = await asyncio.to_thread(
                ingestor.ingest,
                file_path,
                schema_version
            )

        # convert to pipeline result
        result = PipelineResult(
            total_rows=len(ingest_result['data']),
            cleaned_rows=0,
            quarantined_rows=0,
            data=ingest_result['data'],
            errors=ingest_result.get('validation_errors', []),
            metrics={'schema_version': schema_version},
            input_file_path=file_path
        )

        logger.info(f"Ingested {result.total_rows} rows")
        return result

    async def apply_rules(self, input_result: PipelineResult, run_id: str) -> PipelineResult:
        """
        Phase 2: Apply deterministic rules.

        Args:
            input_result: Result from ingest phase
            run_id: Run ID for audit tracking

        Returns:
            Pipeline result with rules applied
        """
        logger.info("Phase 2: Applying deterministic rules")

        # Create a default schema for rules processing
        # This is a simple schema that can handle common data types
        default_schema = Schema(
            version="1.0.0",
            columns=[
                ColumnDefinition(
                    name="Transaction ID",
                    display_name="Transaction ID",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Date",
                    display_name="Date",
                    data_type=ColumnType.DATE,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Account Code",
                    display_name="Account Code",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Account Name",
                    display_name="Account Name",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.LLM_ALLOWED
                ),
                ColumnDefinition(
                    name="Description",
                    display_name="Description",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Debit Amount",
                    display_name="Debit Amount",
                    data_type=ColumnType.DECIMAL,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Credit Amount",
                    display_name="Credit Amount",
                    data_type=ColumnType.DECIMAL,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Department",
                    display_name="Department",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.LLM_ALLOWED
                ),
                ColumnDefinition(
                    name="Reference Number",
                    display_name="Reference Number",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Created By",
                    display_name="Created By",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                )
            ]
        )

        rules_engine = RuleEngine(default_schema)

        # Convert run_id string to UUID
        from uuid import UUID
        run_uuid = UUID(run_id)

        # Apply rules to the entire DataFrame
        transformed_df, patches, audit_events, diff_entries = await asyncio.to_thread(
            rules_engine.apply_rules,
            input_result.data,
            run_uuid
        )

        # Convert audit events to our expected format
        audit_trail = []
        for event in audit_events:
            audit_trail.append({
                'row_id': event.row_uuid,
                'type': 'rule',
                'rule': event.contract_id or 'unknown_rule',
                'field': event.column_name,
                'before': event.before_value,
                'after': event.after_value
            })

        # Save audit events to database if we have a cache (which has db access)
        if self.cache and audit_events:
            try:
                # Convert audit events to the format expected by the database
                db_audit_events = []
                for event in audit_events:
                    db_audit_events.append({
                        'run_id': run_uuid,
                        'row_uuid': event.row_uuid,
                        'source': 'rules',
                        'action': 'modify',
                        'column_name': event.column_name,
                        'before_value': event.before_value,
                        'after_value': event.after_value,
                        'confidence_score': 1.0,  # Rules have 100% confidence
                        'contract_id': event.contract_id
                    })

                # Save to database
                await self.cache.db.create_audit_events_batch(db_audit_events)
                logger.info(f"Saved {len(db_audit_events)} audit events to database")
            except Exception as e:
                logger.warning(f"Failed to save audit events to database: {e}")

        # Count fixes
        fixed_count = len(patches)

        # Update result
        result = PipelineResult(
            total_rows=input_result.total_rows,
            cleaned_rows=fixed_count,
            quarantined_rows=0,
            rules_fixed_count=fixed_count,
            llm_fixed_count=0,
            data=transformed_df,
            errors=input_result.errors,
            audit_trail=audit_trail,
            metrics=input_result.metrics,
            input_file_path=input_result.input_file_path
        )

        result.metrics['rules_applied'] = fixed_count
        logger.info(f"Applied {fixed_count} rule fixes")

        return result

    async def plan_residuals(self,
                            input_result: PipelineResult,
                            llm_columns: List[str]) -> PipelineResult:
        """
        Phase 3: Plan residual errors for LLM processing.

        Args:
            input_result: Result from rules phase
            llm_columns: Columns eligible for LLM processing

        Returns:
            Pipeline result with residual plan
        """
        logger.info("Phase 3: Planning residual errors")

        # Create the same default schema used in rules phase
        default_schema = Schema(
            version="1.0.0",
            columns=[
                ColumnDefinition(
                    name="Transaction ID",
                    display_name="Transaction ID",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Date",
                    display_name="Date",
                    data_type=ColumnType.DATE,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Account Code",
                    display_name="Account Code",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Account Name",
                    display_name="Account Name",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.LLM_ALLOWED
                ),
                ColumnDefinition(
                    name="Description",
                    display_name="Description",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Debit Amount",
                    display_name="Debit Amount",
                    data_type=ColumnType.DECIMAL,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Credit Amount",
                    display_name="Credit Amount",
                    data_type=ColumnType.DECIMAL,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Department",
                    display_name="Department",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.LLM_ALLOWED
                ),
                ColumnDefinition(
                    name="Reference Number",
                    display_name="Reference Number",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Created By",
                    display_name="Created By",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                )
            ]
        )

        planner = ResidualPlanner(default_schema)

        # check cache for existing mappings if available
        cache_hits = 0
        if self.cache:
            # input_result.data is now a DataFrame, so iterate over rows properly
            for index, row in input_result.data.iterrows():
                for column in llm_columns:
                    if column in row:
                        value = row[column]
                        if value and isinstance(value, str):
                            # check cache
                            canonical = await self.cache.get_mapping(column, value)
                            if canonical:
                                cache_hits += 1
                                # Update the DataFrame directly
                                input_result.data.at[index, f"{column}_cached"] = canonical

        # plan residuals
        residual_items, edit_caps = await planner.identify_residuals(
            input_result.data,
            llm_columns
        )

        # update result
        result = PipelineResult(
            total_rows=input_result.total_rows,
            cleaned_rows=input_result.cleaned_rows,
            quarantined_rows=input_result.quarantined_rows,
            rules_fixed_count=input_result.rules_fixed_count,
            llm_fixed_count=input_result.llm_fixed_count,
            data=input_result.data,
            errors=input_result.errors,
            audit_trail=input_result.audit_trail,
            metrics=input_result.metrics,
            input_file_path=input_result.input_file_path
        )

        # Count total residual items for logging
        total_residuals = sum(len(items) for items in residual_items.values())

        result.metrics['residual_items'] = residual_items
        result.metrics['edit_caps'] = edit_caps
        result.metrics['cache_hits'] = cache_hits

        logger.info(f"Identified {total_residuals} residual items, {cache_hits} cache hits")

        return result

    async def process_llm(self,
                         input_result: PipelineResult,
                         dry_run: bool = False) -> PipelineResult:
        """
        Phase 4: Process with LLM.

        Args:
            input_result: Result from planning phase
            dry_run: If True, simulate LLM processing

        Returns:
            Pipeline result with LLM fixes applied
        """
        logger.info(f"Phase 4: Processing with LLM (dry_run={dry_run})")

        # get llm adapter
        if not self.llm_adapter:
            self.llm_adapter = await get_llm_adapter(use_mock=dry_run)

        plan = input_result.metrics.get('residual_plan', {})
        patches = plan.get('patches', [])

        llm_fixed_count = 0
        audit_trail = input_result.audit_trail.copy()

        # process patches
        for patch in patches:
            try:
                # prepare request
                request = {
                    'column': patch['column'],
                    'values': patch['values'],
                    'canonical_values': patch.get('canonical_values', [])
                }

                # call llm
                response = await self.llm_adapter.process(request)

                if response and response.get('success'):
                    mappings = response.get('mappings', {})

                    # apply mappings
                    for value, canonical in mappings.items():
                        # update data
                        for row in input_result.data:
                            if row.get(patch['column']) == value:
                                row[patch['column']] = canonical
                                llm_fixed_count += 1

                                audit_trail.append({
                                    'row_id': row.get('row_id'),
                                    'type': 'llm',
                                    'column': patch['column'],
                                    'before': value,
                                    'after': canonical,
                                    'confidence': response.get('confidence', 0.8)
                                })

                        # store in cache if available
                        if self.cache and not dry_run:
                            await self.cache.store_mapping(
                                patch['column'],
                                value,
                                canonical,
                                confidence=response.get('confidence', 0.8)
                            )

            except Exception as e:
                logger.error(f"LLM processing error: {e}")

        # update result
        result = PipelineResult(
            total_rows=input_result.total_rows,
            cleaned_rows=input_result.cleaned_rows + llm_fixed_count,
            quarantined_rows=input_result.quarantined_rows,
            rules_fixed_count=input_result.rules_fixed_count,
            llm_fixed_count=llm_fixed_count,
            data=input_result.data,
            errors=input_result.errors,
            audit_trail=audit_trail,
            metrics=input_result.metrics,
            input_file_path=input_result.input_file_path
        )

        result.metrics['llm_processed'] = len(patches)
        result.metrics['llm_fixed'] = llm_fixed_count

        logger.info(f"LLM fixed {llm_fixed_count} values")

        return result

    async def validate_final(self, input_result: PipelineResult) -> PipelineResult:
        """
        Phase 5: Final validation and quarantine.

        Args:
            input_result: Result from LLM phase

        Returns:
            Final pipeline result
        """
        logger.info("Phase 5: Final validation")

        # Create the same default schema used in previous phases
        default_schema = Schema(
            version="1.0.0",
            columns=[
                ColumnDefinition(
                    name="Transaction ID",
                    display_name="Transaction ID",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Date",
                    display_name="Date",
                    data_type=ColumnType.DATE,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Account Code",
                    display_name="Account Code",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Account Name",
                    display_name="Account Name",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.LLM_ALLOWED
                ),
                ColumnDefinition(
                    name="Description",
                    display_name="Description",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Debit Amount",
                    display_name="Debit Amount",
                    data_type=ColumnType.DECIMAL,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Credit Amount",
                    display_name="Credit Amount",
                    data_type=ColumnType.DECIMAL,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Department",
                    display_name="Department",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.LLM_ALLOWED
                ),
                ColumnDefinition(
                    name="Reference Number",
                    display_name="Reference Number",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Created By",
                    display_name="Created By",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                )
            ]
        )

        quarantine_mgr = QuarantineManager(default_schema)

        # validate all rows using the DataFrame directly
        clean_df, quarantine_df, quarantine_rows, validation_stats = await asyncio.to_thread(
            quarantine_mgr.validate_and_quarantine,
            input_result.data
        )

        # update result
        result = PipelineResult(
            total_rows=input_result.total_rows,
            cleaned_rows=len(clean_df),
            quarantined_rows=len(quarantine_df),
            rules_fixed_count=input_result.rules_fixed_count,
            llm_fixed_count=input_result.llm_fixed_count,
            data=clean_df,
            errors=quarantine_rows,
            audit_trail=input_result.audit_trail,
            metrics=input_result.metrics,
            input_file_path=input_result.input_file_path
        )

        # Add validation statistics to metrics
        result.metrics.update(validation_stats)

        # Store the quarantine DataFrame for artifact generation
        result.quarantine_df = quarantine_df

        logger.info(f"Final: {result.cleaned_rows} clean, {result.quarantined_rows} quarantined")

        return result

    async def generate_artifacts(self,
                                run_id: str,
                                result: PipelineResult) -> Dict[str, str]:
        """
        Phase 6: Generate and store artifacts.

        Args:
            run_id: Run identifier
            result: Final pipeline result

        Returns:
            Map of artifact types to storage paths
        """
        logger.info("Phase 6: Generating artifacts")

        # Create the same default schema used in previous phases
        default_schema = Schema(
            version="1.0.0",
            columns=[
                ColumnDefinition(
                    name="Transaction ID",
                    display_name="Transaction ID",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Date",
                    display_name="Date",
                    data_type=ColumnType.DATE,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Account Code",
                    display_name="Account Code",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Account Name",
                    display_name="Account Name",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.LLM_ALLOWED
                ),
                ColumnDefinition(
                    name="Description",
                    display_name="Description",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Debit Amount",
                    display_name="Debit Amount",
                    data_type=ColumnType.DECIMAL,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Credit Amount",
                    display_name="Credit Amount",
                    data_type=ColumnType.DECIMAL,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Department",
                    display_name="Department",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.LLM_ALLOWED
                ),
                ColumnDefinition(
                    name="Reference Number",
                    display_name="Reference Number",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="Created By",
                    display_name="Created By",
                    data_type=ColumnType.STRING,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                )
            ]
        )

        # Convert run_id string to UUID if needed
        from uuid import UUID
        run_uuid = UUID(run_id) if isinstance(run_id, str) else run_id

        artifact_mgr = ArtifactManager(
            run_id=run_uuid,
            schema=default_schema,
            minio_client=self.storage.client if hasattr(self.storage, 'client') else None
        )

        artifacts = {}

        # Prepare data for artifact generation
        try:
            # Create manifest and metrics objects
            from core.models import Manifest, Metrics, RunOptions
            from datetime import datetime
            import hashlib

            # Create default run options
            run_options = RunOptions(
                use_inferred=False,
                dry_run=False,
                llm_columns=["Department", "Account Name"],
                edit_cap_pct=20,
                confidence_floor=0.80,
                row_limit=50000,
                batch_size=15
            )

            manifest = Manifest(
                run_id=run_uuid,
                run_seq=1,  # Simple sequential number
                schema_version="1.0.0",
                model_id="openai/gpt-4",
                prompt_version="1.0.0",
                engine_version="0.1.0",
                input_file_name="input.csv",
                input_file_hash=result.metrics.get('input_hash', 'unknown'),
                input_row_count=result.total_rows,
                options=run_options,
                seed=42,
                temperature=0.0,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                idempotency_key=hashlib.sha256(f"{run_id}_1.0.0".encode()).hexdigest()
            )

            metrics = Metrics(
                run_id=run_uuid,
                total_rows=result.total_rows,
                total_cells=result.total_rows * 10,  # Estimate based on 10 columns
                cells_validated=result.total_rows * 10,
                cells_modified=result.rules_fixed_count + result.llm_fixed_count,
                cells_quarantined=result.quarantined_rows * 10,
                rules_fixed_count=result.rules_fixed_count,
                llm_fixed_count=result.llm_fixed_count,
                cache_fixed_count=0,
                rules_duration_ms=1000,  # Placeholder timing
                llm_duration_ms=500,
                validation_duration_ms=300,
                total_duration_ms=2000,
                llm_calls_count=0,
                llm_tokens_used=0,
                llm_cost_estimate=None,
                cache_hits=0,
                cache_misses=0
            )

            # Extract quarantine data if available
            import pandas as pd
            quarantine_df = getattr(result, 'quarantine_df', pd.DataFrame())
            quarantine_rows = result.errors if result.errors else []
            quarantine_stats = result.metrics if 'quarantine_stats' in result.metrics else {}

            # Extract audit events and diffs from metrics
            audit_events = []
            diffs = []

            # Store original input file as input.csv if available
            if result.input_file_path and os.path.exists(result.input_file_path):
                try:
                    import shutil
                    from core.config import settings
                    # Copy input file to artifacts directory with standardized name
                    input_storage_path = f"artifacts/{run_id}/input.csv"

                    # Use storage backend to store the input file
                    with open(result.input_file_path, 'rb') as f:
                        input_content = f.read()

                    # Calculate hash for the input file
                    input_hash = hashlib.sha256(input_content).hexdigest()

                    # Store in MinIO/storage backend
                    if hasattr(self.storage, 'client'):  # MinIO client
                        bucket_name = settings.MINIO_BUCKET  # Use same bucket as other artifacts
                        self.storage.client.put_object(
                            bucket_name,
                            input_storage_path,
                            io.BytesIO(input_content),
                            len(input_content),
                            content_type='text/csv'
                        )

                        artifacts['input.csv'] = {
                            'storage_path': input_storage_path,
                            'content_hash': input_hash,
                            'size_bytes': len(input_content),
                            'row_count': result.total_rows
                        }

                        logger.info(f"Stored input file as artifacts/{run_id}/input.csv")

                except Exception as e:
                    logger.warning(f"Failed to store input file: {e}")

            # Generate all artifacts
            artifact_metadata = await asyncio.to_thread(
                artifact_mgr.generate_all_artifacts,
                result.data,  # cleaned_df
                quarantine_df,
                manifest,
                metrics,
                audit_events,
                diffs,
                quarantine_rows,
                quarantine_stats
            )

            artifacts.update(artifact_metadata)

            # Register artifacts in database if artifact store is available
            if self.artifact_store and artifacts:
                for artifact_name, artifact_info in artifacts.items():
                    if isinstance(artifact_info, dict) and 'storage_path' in artifact_info:
                        # Register each artifact in the database
                        await self.artifact_store.register_artifact(
                            run_id=run_id,
                            artifact_type=artifact_name.replace('.csv', '').replace('.json', '').replace('.md', ''),
                            file_name=artifact_name,
                            content_hash=artifact_info.get('content_hash', 'unknown'),
                            storage_path=artifact_info['storage_path'],
                            size_bytes=artifact_info.get('size_bytes', 0),
                            mime_type=self._get_mime_type(artifact_name),
                            row_count=artifact_info.get('row_count', 0)
                        )

        except Exception as e:
            logger.error(f"Error generating artifacts: {e}")
            artifacts['error'] = str(e)

        logger.info(f"Generated {len(artifacts)} artifacts")

        return artifacts

    def _get_mime_type(self, filename: str) -> str:
        """Get MIME type based on file extension."""
        if filename.endswith('.csv'):
            return 'text/csv'
        elif filename.endswith('.json'):
            return 'application/json'
        elif filename.endswith('.md'):
            return 'text/markdown'
        elif filename.endswith('.ndjson'):
            return 'application/x-ndjson'
        else:
            return 'application/octet-stream'
