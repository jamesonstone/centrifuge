"""Artifact generation and storage management."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import pandas as pd
import structlog
from minio import Minio
from minio.error import S3Error

from core.models import (
    Manifest, Metrics, AuditEvent, DiffEntry,
    Patch, QuarantineRow, Schema
)
from core.config import settings

logger = structlog.get_logger()


class ArtifactManager:
    """Manages generation and storage of all pipeline artifacts."""

    def __init__(
        self,
        run_id: UUID,
        schema: Schema,
        minio_client: Optional[Minio] = None
    ):
        """
        Initialize artifact manager.

        Args:
            run_id: Run identifier
            schema: Schema used for processing
            minio_client: MinIO client for artifact storage
        """
        self.run_id = run_id
        self.schema = schema
        self.minio_client = minio_client
        self.artifacts: Dict[str, Dict[str, Any]] = {}

    def generate_all_artifacts(
        self,
        cleaned_df: pd.DataFrame,
        quarantine_df: pd.DataFrame,
        manifest: Manifest,
        metrics: Metrics,
        audit_events: List[AuditEvent],
        diffs: List[DiffEntry],
        quarantine_rows: List[QuarantineRow],
        quarantine_stats: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate all required artifacts.

        Args:
            cleaned_df: Cleaned data
            quarantine_df: Quarantined rows
            manifest: Run manifest
            metrics: Run metrics
            audit_events: All audit events
            diffs: All diff entries
            quarantine_rows: Quarantine row details
            quarantine_stats: Quarantine statistics

        Returns:
            Dictionary of artifact metadata
        """
        logger.info("Generating artifacts", run_id=str(self.run_id))

        # Generate each artifact
        artifacts = {}

        # 1. cleaned.csv
        cleaned_path = self._generate_cleaned_csv(cleaned_df)
        artifacts['cleaned.csv'] = cleaned_path

        # 2. errors.csv
        if len(quarantine_df) > 0:
            errors_path = self._generate_errors_csv(quarantine_df, quarantine_rows)
            artifacts['errors.csv'] = errors_path

        # 3. diff.csv
        if diffs:
            diff_path = self._generate_diff_csv(diffs)
            artifacts['diff.csv'] = diff_path

        # 4. audit.ndjson
        if audit_events:
            audit_path = self._generate_audit_ndjson(audit_events)
            artifacts['audit.ndjson'] = audit_path

        # 5. manifest.json
        manifest_path = self._generate_manifest_json(manifest)
        artifacts['manifest.json'] = manifest_path

        # 6. metrics.json
        metrics_path = self._generate_metrics_json(metrics)
        artifacts['metrics.json'] = metrics_path

        # 7. summary.md
        summary_path = self._generate_summary_md(
            cleaned_df, quarantine_df, metrics,
            audit_events, diffs, quarantine_stats
        )
        artifacts['summary.md'] = summary_path

        self.artifacts = artifacts

        logger.info("Artifacts generated",
                   run_id=str(self.run_id),
                   artifact_count=len(artifacts))

        return artifacts

    def _generate_cleaned_csv(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate cleaned.csv artifact."""
        content = df.to_csv(index=False)
        return self._save_artifact('cleaned.csv', content)

    def _generate_errors_csv(
        self,
        quarantine_df: pd.DataFrame,
        quarantine_rows: List[QuarantineRow]
    ) -> Dict[str, Any]:
        """Generate errors.csv artifact with enriched error information."""
        # Build enriched error dataframe
        error_data = []

        for qrow in quarantine_rows:
            row_dict = {
                'row_number': qrow.row_number,
                'error_category': qrow.error_category.value,
                'error_count': len(qrow.error_details),
                'error_summary': '; '.join(qrow.error_details[:3])
            }

            # Add original data
            for key, value in qrow.original_data.items():
                if key not in ['row_uuid', 'row_number']:
                    row_dict[key] = value

            # Add attempted fixes summary
            if qrow.attempted_fixes:
                row_dict['attempted_fixes'] = f"{len(qrow.attempted_fixes)} attempted"
            else:
                row_dict['attempted_fixes'] = 'None'

            error_data.append(row_dict)

        # Create DataFrame - handle empty case
        if error_data:
            errors_df = pd.DataFrame(error_data)

            # Reorder columns
            metadata_cols = ['row_number', 'error_category', 'error_count', 'error_summary', 'attempted_fixes']
            data_cols = [col for col in errors_df.columns if col not in metadata_cols]
            ordered_cols = metadata_cols + data_cols
            errors_df = errors_df[ordered_cols]
        else:
            # Create empty DataFrame with expected structure
            metadata_cols = ['row_number', 'error_category', 'error_count', 'error_summary', 'attempted_fixes']
            errors_df = pd.DataFrame(columns=metadata_cols)

        content = errors_df.to_csv(index=False)
        return self._save_artifact('errors.csv', content)

    def _generate_diff_csv(self, diffs: List[DiffEntry]) -> Dict[str, Any]:
        """Generate diff.csv artifact."""
        diff_data = []
        for diff in diffs:
            diff_data.append({
                'row_uuid': str(diff.row_uuid),
                'row_number': diff.row_number,
                'column': diff.column_name,
                'before_value': diff.before_value,
                'after_value': diff.after_value,
                'source': diff.source.value,
                'reason': diff.reason,
                'confidence': diff.confidence,
                'timestamp': diff.timestamp.isoformat() if diff.timestamp else None
            })

        diff_df = pd.DataFrame(diff_data)
        content = diff_df.to_csv(index=False)
        return self._save_artifact('diff.csv', content)

    def _generate_audit_ndjson(self, audit_events: List[AuditEvent]) -> Dict[str, Any]:
        """Generate audit.ndjson artifact."""
        lines = []
        for event in audit_events:
            event_dict = {
                'event_id': str(event.id),
                'run_id': str(event.run_id),
                'timestamp': event.created_at.isoformat(),
                'event_type': event.source.value,
                'phase': 'rules',  # Default phase for rules engine events
                'row_uuid': str(event.row_uuid) if event.row_uuid else None,
                'column_name': event.column_name,
                'before_value': event.before_value,
                'after_value': event.after_value,
                'source': event.source.value if event.source else None,
                'rule_id': event.rule_id,
                'contract_id': event.contract_id,
                'reason': event.reason,
                'confidence': event.confidence,
                'metadata': {
                    'rule_id': event.rule_id,
                    'contract_id': event.contract_id,
                    'reason': event.reason,
                    'confidence': event.confidence
                }
            }
            lines.append(json.dumps(event_dict, default=str))

        content = '\n'.join(lines)
        return self._save_artifact('audit.ndjson', content)

    def _generate_manifest_json(self, manifest: Manifest) -> Dict[str, Any]:
        """Generate manifest.json artifact."""
        manifest_dict = {
            'run_id': str(manifest.run_id),
            'created_at': manifest.started_at.isoformat(),
            'input_hash': manifest.input_file_hash,
            'schema_version': manifest.schema_version,
            'model_version': manifest.model_id,
            'prompt_version': manifest.prompt_version,
            'recipe_version': manifest.engine_version,
            'seed': manifest.seed,
            'options': manifest.options.model_dump() if manifest.options else {},
            'idempotency_key': manifest.idempotency_key
        }

        content = json.dumps(manifest_dict, indent=2, default=str)
        return self._save_artifact('manifest.json', content)

    def _generate_metrics_json(self, metrics: Metrics) -> Dict[str, Any]:
        """Generate metrics.json artifact."""
        # Calculate derived metrics
        quarantined_rows = metrics.cells_quarantined // max(1, (metrics.total_cells // metrics.total_rows)) if metrics.total_cells > 0 else 0
        clean_rows = metrics.total_rows - quarantined_rows
        success_rate = clean_rows / metrics.total_rows if metrics.total_rows > 0 else 0
        quarantine_rate = quarantined_rows / metrics.total_rows if metrics.total_rows > 0 else 0

        rules_fixed_percentage = (metrics.rules_fixed_count / metrics.total_rows) if metrics.total_rows > 0 else 0
        llm_fixed_percentage = (metrics.llm_fixed_count / metrics.total_rows) if metrics.total_rows > 0 else 0

        metrics_dict = {
            'run_id': str(metrics.run_id),
            'total_rows': metrics.total_rows,
            'clean_rows': clean_rows,
            'quarantined_rows': quarantined_rows,
            'success_rate': success_rate,
            'quarantine_rate': quarantine_rate,

            'rules_applied': metrics.rules_fixed_count,  # Use rules_fixed_count as proxy
            'rules_fixed_count': metrics.rules_fixed_count,
            'rules_fixed_percentage': rules_fixed_percentage,

            'llm_calls_made': metrics.llm_calls_count,
            'llm_fixed_count': metrics.llm_fixed_count,
            'llm_fixed_percentage': llm_fixed_percentage,
            'llm_tokens_used': metrics.llm_tokens_used,
            'llm_cost_estimate': float(metrics.llm_cost_estimate) if metrics.llm_cost_estimate else 0.0,

            'cache_hits': metrics.cache_hits,
            'cache_misses': metrics.cache_misses,
            'cache_hit_rate': metrics.cache_hit_ratio,

            'phase_timings': {
                'rules_duration_ms': metrics.rules_duration_ms,
                'llm_duration_ms': metrics.llm_duration_ms,
                'validation_duration_ms': metrics.validation_duration_ms
            },
            'total_duration_seconds': metrics.total_duration_ms / 1000.0,

            'errors_by_category': {
                'validation_failures': metrics.validation_failures,
                'llm_contract_failures': metrics.llm_contract_failures,
                'low_confidence_count': metrics.low_confidence_count,
                'edit_cap_exceeded_count': metrics.edit_cap_exceeded_count,
                'parse_errors': metrics.parse_errors
            },
            'errors_by_column': metrics.column_stats,

            'edit_caps': {},  # Not available in current model
            'confidence_distribution': {}  # Not available in current model
        }

        content = json.dumps(metrics_dict, indent=2, default=str)
        return self._save_artifact('metrics.json', content)

    def _generate_summary_md(
        self,
        cleaned_df: pd.DataFrame,
        quarantine_df: pd.DataFrame,
        metrics: Metrics,
        audit_events: List[AuditEvent],
        diffs: List[DiffEntry],
        quarantine_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate summary.md artifact."""
        # Calculate derived metrics
        quarantined_rows = metrics.cells_quarantined // max(1, (metrics.total_cells // metrics.total_rows)) if metrics.total_cells > 0 else 0
        clean_rows = metrics.total_rows - quarantined_rows
        success_rate = clean_rows / metrics.total_rows if metrics.total_rows > 0 else 0
        quarantine_rate = quarantined_rows / metrics.total_rows if metrics.total_rows > 0 else 0

        rules_fixed_percentage = (metrics.rules_fixed_count / metrics.total_rows) if metrics.total_rows > 0 else 0
        llm_fixed_percentage = (metrics.llm_fixed_count / metrics.total_rows) if metrics.total_rows > 0 else 0
        cache_hit_rate = metrics.cache_hit_ratio if metrics.cache_hit_ratio else 0

        # Calculate additional statistics
        rules_changes = [d for d in diffs if d.source.value == 'rule']
        llm_changes = [d for d in diffs if d.source.value == 'llm']
        cache_changes = [d for d in diffs if d.source.value == 'cache']

        # Build markdown content
        lines = [
            f"# Centrifuge Run Summary",
            f"**Run ID:** {self.run_id}",
            f"**Date:** {datetime.now().isoformat()}",
            f"**Schema Version:** {self.schema.version}",
            "",
            "## Overview",
            f"- **Total Rows Processed:** {metrics.total_rows:,}",
            f"- **Clean Rows:** {clean_rows:,} ({success_rate:.1%})",
            f"- **Quarantined Rows:** {quarantined_rows:,} ({quarantine_rate:.1%})",
            "",
            "## Data Quality Improvements",
            f"### Rules Engine",
            f"- **Changes Applied:** {len(rules_changes):,}",
            f"- **Rows Fixed:** {metrics.rules_fixed_count:,}",
            f"- **Row Impact Rate:** {rules_fixed_percentage:.1%}",
            "",
            f"### LLM Processing",
            f"- **API Calls:** {metrics.llm_calls_count:,}",
            f"- **Changes Applied:** {len(llm_changes):,}",
            f"- **Rows Fixed:** {metrics.llm_fixed_count:,}",
            f"- **Row Impact Rate:** {llm_fixed_percentage:.1%}",
            f"- **Tokens Used:** {metrics.llm_tokens_used:,}" if metrics.llm_tokens_used else "- **Tokens Used:** N/A",
            f"- **Estimated Cost:** ${float(metrics.llm_cost_estimate):.2f}" if metrics.llm_cost_estimate else "- **Estimated Cost:** N/A",
            "",
            f"### Cache Performance",
            f"- **Cache Hits:** {metrics.cache_hits:,}",
            f"- **Cache Misses:** {metrics.cache_misses:,}",
            f"- **Hit Rate:** {cache_hit_rate:.1%}",
            f"- **Changes from Cache:** {len(cache_changes):,}",
            "",
            "## Quarantine Analysis",
        ]

        if quarantine_stats and quarantine_stats.get('errors_by_category'):
            lines.append("### Error Categories")
            for category, count in quarantine_stats['errors_by_category'].items():
                percentage = count / quarantined_rows * 100 if quarantined_rows > 0 else 0
                lines.append(f"- **{category}:** {count} ({percentage:.1f}%)")
            lines.append("")

        if quarantine_stats and quarantine_stats.get('top_error_messages'):
            lines.append("### Top Error Messages")
            for i, error_info in enumerate(quarantine_stats['top_error_messages'][:5], 1):
                lines.append(f"{i}. {error_info['message']} ({error_info['count']} occurrences)")
            lines.append("")

        # Phase timings
        phase_timings = {
            'Rules Engine': metrics.rules_duration_ms / 1000.0,
            'LLM Processing': metrics.llm_duration_ms / 1000.0,
            'Validation': metrics.validation_duration_ms / 1000.0
        }
        total_duration_seconds = metrics.total_duration_ms / 1000.0

        if any(duration > 0 for duration in phase_timings.values()):
            lines.extend([
                "## Performance Metrics",
                "### Phase Timings"
            ])
            for phase, duration in phase_timings.items():
                lines.append(f"- **{phase}:** {duration:.2f}s")
            lines.append(f"- **Total Duration:** {total_duration_seconds:.2f}s")
            lines.append("")

        # Column-specific issues (use column_stats if available)
        if metrics.column_stats:
            lines.append("### Issues by Column")
            # Convert column_stats to error counts per column
            sorted_columns = sorted(
                [(col, sum(stats.values())) for col, stats in metrics.column_stats.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for column, count in sorted_columns:
                if count > 0:
                    lines.append(f"- **{column}:** {count} issues")
            lines.append("")

        # Edit caps - not available in current metrics model
        # TODO: Implement edit cap tracking if needed

        # Confidence distribution - not available in current metrics model
        # TODO: Implement confidence distribution tracking if needed

        # Summary
        lines.extend([
            "## Summary",
            f"The pipeline successfully processed {success_rate:.1%} of input rows. "
            f"Rules-based cleaning affected {rules_fixed_percentage:.1%} of rows, "
            f"while LLM processing affected {llm_fixed_percentage:.1%} of rows. "
        ])

        if quarantined_rows > 0:
            lines.append(
                f"{quarantined_rows:,} rows were quarantined for manual review. "
                f"The most common issue was {list(quarantine_stats.get('errors_by_category', {}).keys())[0] if quarantine_stats.get('errors_by_category') else 'validation failures'}."
            )

        lines.extend([
            "",
            "## Artifacts",
            "- `cleaned.csv` - Successfully processed data",
            "- `errors.csv` - Quarantined rows with error details" if quarantined_rows > 0 else "",
            "- `diff.csv` - All data transformations",
            "- `audit.ndjson` - Detailed audit trail",
            "- `manifest.json` - Run configuration and versions",
            "- `metrics.json` - Detailed performance metrics",
            "- `summary.md` - This summary report",
            "",
            "---",
            f"*Generated by Centrifuge v{self.schema.version}*"
        ])

        content = '\n'.join(line for line in lines if line is not None)
        return self._save_artifact('summary.md', content)

    def _save_artifact(
        self,
        artifact_name: str,
        content: str
    ) -> Dict[str, Any]:
        """
        Save artifact and return metadata.

        Args:
            artifact_name: Name of artifact
            content: Artifact content

        Returns:
            Artifact metadata including hash and storage path
        """
        # Calculate content hash
        content_bytes = content.encode('utf-8')
        content_hash = hashlib.sha256(content_bytes).hexdigest()

        # Create metadata
        metadata = {
            'name': artifact_name,
            'content_hash': content_hash,
            'size_bytes': len(content_bytes),
            'created_at': datetime.now().isoformat()
        }

        # Store to MinIO if client available
        if self.minio_client:
            try:
                # Run-based path for easy lookup
                object_path = f"artifacts/{self.run_id}/{artifact_name}"

                # Upload to MinIO
                from io import BytesIO
                self.minio_client.put_object(
                    settings.MINIO_BUCKET,
                    object_path,
                    BytesIO(content_bytes),
                    len(content_bytes)
                )

                metadata['storage_path'] = object_path
                metadata['storage_type'] = 'minio'

                logger.info("Artifact stored to MinIO",
                           artifact=artifact_name,
                           path=object_path,
                           hash=content_hash)

            except S3Error as e:
                logger.error("Failed to store artifact to MinIO",
                            artifact=artifact_name,
                            error=str(e))
                # Fall back to local storage
                metadata['storage_type'] = 'local'
        else:
            # Local storage fallback
            local_dir = Path(f"artifacts/{self.run_id}")
            local_dir.mkdir(parents=True, exist_ok=True)

            local_path = local_dir / artifact_name
            local_path.write_text(content)

            metadata['storage_path'] = str(local_path)
            metadata['storage_type'] = 'local'

            logger.info("Artifact stored locally",
                       artifact=artifact_name,
                       path=str(local_path),
                       hash=content_hash)

        return metadata

    def get_artifact_summary(self) -> Dict[str, Any]:
        """Get summary of all generated artifacts."""
        return {
            'run_id': str(self.run_id),
            'artifact_count': len(self.artifacts),
            'artifacts': self.artifacts,
            'total_size_bytes': sum(
                a.get('size_bytes', 0)
                for a in self.artifacts.values()
            )
        }

    def store_artifact_metadata_to_db(
        self,
        db_connection: Any
    ) -> None:
        """
        Store artifact metadata to database.

        Args:
            db_connection: Database connection
        """
        # This would store artifact metadata to Postgres
        # Including URIs, hashes, and run_id links
        # Implementation deferred for actual database integration
        pass
