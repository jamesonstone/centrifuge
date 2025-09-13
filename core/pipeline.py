"""
Data processing pipeline for Centrifuge.
Orchestrates the flow through all processing phases.
"""

import os
import logging
import hashlib
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

from core.ingest import DataIngestor
from core.rules import RulesEngine
from core.residual_planner import ResidualPlanner
from core.llm_client import get_llm_adapter
from core.quarantine import QuarantineManager
from core.artifacts import ArtifactManager
from core.database import CanonicalMappingCache, ArtifactStore
from core.storage import StorageBackend

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
            metrics={'schema_version': schema_version}
        )
        
        logger.info(f"Ingested {result.total_rows} rows")
        return result
    
    async def apply_rules(self, input_result: PipelineResult) -> PipelineResult:
        """
        Phase 2: Apply deterministic rules.
        
        Args:
            input_result: Result from ingest phase
            
        Returns:
            Pipeline result with rules applied
        """
        logger.info("Phase 2: Applying deterministic rules")
        
        rules_engine = RulesEngine()
        
        # apply rules to each row
        fixed_count = 0
        audit_trail = []
        
        for row in input_result.data:
            fixes = await asyncio.to_thread(rules_engine.apply_rules, row)
            
            if fixes:
                fixed_count += len(fixes)
                for fix in fixes:
                    audit_trail.append({
                        'row_id': row.get('row_id'),
                        'type': 'rule',
                        'rule': fix['rule'],
                        'field': fix['field'],
                        'before': fix['before'],
                        'after': fix['after']
                    })
        
        # update result
        result = PipelineResult(
            total_rows=input_result.total_rows,
            cleaned_rows=fixed_count,
            quarantined_rows=0,
            rules_fixed_count=fixed_count,
            llm_fixed_count=0,
            data=input_result.data,
            errors=input_result.errors,
            audit_trail=audit_trail,
            metrics=input_result.metrics
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
        
        planner = ResidualPlanner()
        
        # check cache for existing mappings if available
        cache_hits = 0
        if self.cache:
            for row in input_result.data:
                for column in llm_columns:
                    value = row.get(column)
                    if value and isinstance(value, str):
                        # check cache
                        canonical = await self.cache.get_mapping(column, value)
                        if canonical:
                            cache_hits += 1
                            row[f"{column}_cached"] = canonical
        
        # plan residuals
        plan = await asyncio.to_thread(
            planner.plan,
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
            metrics=input_result.metrics
        )
        
        result.metrics['residual_plan'] = plan
        result.metrics['cache_hits'] = cache_hits
        
        logger.info(f"Planned {len(plan.get('patches', []))} LLM patches, {cache_hits} cache hits")
        
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
            metrics=input_result.metrics
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
        
        quarantine_mgr = QuarantineManager()
        
        # validate all rows
        clean_rows = []
        quarantined_rows = []
        error_breakdown = {}
        
        for row in input_result.data:
            validation_result = await asyncio.to_thread(
                quarantine_mgr.validate_row, 
                row
            )
            
            if validation_result['valid']:
                clean_rows.append(row)
            else:
                quarantined_rows.append({
                    'row': row,
                    'errors': validation_result['errors']
                })
                
                # track error types
                for error in validation_result['errors']:
                    error_type = error.get('type', 'unknown')
                    error_breakdown[error_type] = error_breakdown.get(error_type, 0) + 1
        
        # update result
        result = PipelineResult(
            total_rows=input_result.total_rows,
            cleaned_rows=len(clean_rows),
            quarantined_rows=len(quarantined_rows),
            rules_fixed_count=input_result.rules_fixed_count,
            llm_fixed_count=input_result.llm_fixed_count,
            error_breakdown=error_breakdown,
            data=clean_rows,
            errors=quarantined_rows,
            audit_trail=input_result.audit_trail,
            metrics=input_result.metrics
        )
        
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
        
        artifact_mgr = ArtifactManager(
            run_id=run_id,
            storage=self.storage
        )
        
        artifacts = {}
        
        # generate cleaned data CSV
        if result.cleaned_rows > 0:
            cleaned_path = await artifact_mgr.save_cleaned_data(result.data)
            artifacts['cleaned'] = cleaned_path
            
            # register in database if available
            if self.artifact_store:
                await self.artifact_store.register_artifact(
                    run_id=run_id,
                    artifact_type='cleaned',
                    file_name='cleaned.csv',
                    content_hash=hashlib.sha256(
                        json.dumps(result.data).encode()
                    ).hexdigest(),
                    storage_path=cleaned_path,
                    size_bytes=len(json.dumps(result.data)),
                    mime_type='text/csv',
                    row_count=result.cleaned_rows
                )
        
        # generate error report
        if result.quarantined_rows > 0:
            errors_path = await artifact_mgr.save_errors(result.errors)
            artifacts['errors'] = errors_path
            
            if self.artifact_store:
                await self.artifact_store.register_artifact(
                    run_id=run_id,
                    artifact_type='errors',
                    file_name='errors.json',
                    content_hash=hashlib.sha256(
                        json.dumps(result.errors).encode()
                    ).hexdigest(),
                    storage_path=errors_path,
                    size_bytes=len(json.dumps(result.errors)),
                    mime_type='application/json',
                    row_count=result.quarantined_rows
                )
        
        # generate audit trail
        if result.audit_trail:
            audit_path = await artifact_mgr.save_audit_trail(result.audit_trail)
            artifacts['audit'] = audit_path
            
            if self.artifact_store:
                await self.artifact_store.register_artifact(
                    run_id=run_id,
                    artifact_type='audit',
                    file_name='audit.json',
                    content_hash=hashlib.sha256(
                        json.dumps(result.audit_trail).encode()
                    ).hexdigest(),
                    storage_path=audit_path,
                    size_bytes=len(json.dumps(result.audit_trail)),
                    mime_type='application/json'
                )
        
        # generate summary report
        summary = {
            'run_id': run_id,
            'timestamp': datetime.utcnow().isoformat(),
            'total_rows': result.total_rows,
            'cleaned_rows': result.cleaned_rows,
            'quarantined_rows': result.quarantined_rows,
            'rules_fixed': result.rules_fixed_count,
            'llm_fixed': result.llm_fixed_count,
            'error_breakdown': result.error_breakdown,
            'metrics': result.metrics
        }
        
        summary_path = await artifact_mgr.save_summary(summary)
        artifacts['summary'] = summary_path
        
        if self.artifact_store:
            await self.artifact_store.register_artifact(
                run_id=run_id,
                artifact_type='summary',
                file_name='summary.json',
                content_hash=hashlib.sha256(
                    json.dumps(summary).encode()
                ).hexdigest(),
                storage_path=summary_path,
                size_bytes=len(json.dumps(summary)),
                mime_type='application/json'
            )
        
        logger.info(f"Generated {len(artifacts)} artifacts")
        
        return artifacts