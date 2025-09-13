"""Apply patches with preconditions and comprehensive audit tracking."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import pandas as pd
import structlog

from core.models import (
    Patch, AuditEvent, DiffEntry, SourceType,
    ErrorCategory
)
from core.db_models import DatabaseManager
from core.config import settings

logger = structlog.get_logger()


class PatchApplicationResult:
    """Result of applying a patch."""
    
    def __init__(
        self,
        patch: Patch,
        applied: bool,
        reason: str,
        actual_value: Optional[str] = None
    ):
        self.patch = patch
        self.applied = applied
        self.reason = reason
        self.actual_value = actual_value  # Value found when precondition failed


class PatchApplicator:
    """Applies patches with precondition checking and audit logging."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize patch applicator.
        
        Args:
            db_manager: Database manager for audit persistence
        """
        self.db = db_manager
        self.application_results: List[PatchApplicationResult] = []
        self.audit_events: List[AuditEvent] = []
        self.diff_entries: List[DiffEntry] = []
        
    def apply_patches(
        self,
        df: pd.DataFrame,
        patches: List[Patch],
        run_id: UUID,
        enforce_preconditions: bool = True
    ) -> Tuple[pd.DataFrame, List[AuditEvent], List[DiffEntry], Dict[str, Any]]:
        """
        Apply patches to DataFrame with precondition checking.
        
        Args:
            df: DataFrame to update
            patches: List of patches to apply
            run_id: Run ID for audit
            enforce_preconditions: Whether to check preconditions
            
        Returns:
            Tuple of (updated_df, audit_events, diff_entries, statistics)
        """
        logger.info("Starting patch application", 
                   patches_count=len(patches),
                   enforce_preconditions=enforce_preconditions)
        
        # Reset tracking
        self.application_results = []
        self.audit_events = []
        self.diff_entries = []
        
        # Statistics
        stats = {
            'total_patches': len(patches),
            'applied': 0,
            'skipped_precondition': 0,
            'skipped_not_found': 0,
            'skipped_low_confidence': 0,
            'by_source': {
                'rule': 0,
                'llm': 0,
                'cache': 0,
                'human': 0
            }
        }
        
        # Group patches by row for efficiency
        patches_by_row = self._group_patches_by_row(patches)
        
        # Apply patches
        for row_uuid, row_patches in patches_by_row.items():
            # Find row in DataFrame
            row_mask = df['row_uuid'] == str(row_uuid)
            if not row_mask.any():
                logger.warning("Row not found for patch", row_uuid=row_uuid)
                for patch in row_patches:
                    result = PatchApplicationResult(
                        patch=patch,
                        applied=False,
                        reason="Row not found in DataFrame"
                    )
                    self.application_results.append(result)
                    stats['skipped_not_found'] += 1
                continue
            
            row_idx = df[row_mask].index[0]
            
            # Apply each patch for this row
            for patch in row_patches:
                result = self._apply_single_patch(
                    df, row_idx, patch, run_id,
                    enforce_preconditions
                )
                self.application_results.append(result)
                
                if result.applied:
                    stats['applied'] += 1
                    stats['by_source'][patch.source.value] += 1
                elif 'precondition' in result.reason.lower():
                    stats['skipped_precondition'] += 1
                elif 'confidence' in result.reason.lower():
                    stats['skipped_low_confidence'] += 1
                else:
                    stats['skipped_not_found'] += 1
        
        # Log summary
        logger.info("Patch application complete",
                   applied=stats['applied'],
                   skipped=stats['total_patches'] - stats['applied'],
                   precondition_failures=stats['skipped_precondition'])
        
        return df, self.audit_events, self.diff_entries, stats
    
    def _group_patches_by_row(self, patches: List[Patch]) -> Dict[UUID, List[Patch]]:
        """Group patches by row UUID for efficient processing."""
        grouped = {}
        for patch in patches:
            if patch.row_uuid not in grouped:
                grouped[patch.row_uuid] = []
            grouped[patch.row_uuid].append(patch)
        return grouped
    
    def _apply_single_patch(
        self,
        df: pd.DataFrame,
        row_idx: int,
        patch: Patch,
        run_id: UUID,
        enforce_preconditions: bool
    ) -> PatchApplicationResult:
        """
        Apply a single patch to a row.
        
        Args:
            df: DataFrame to update
            row_idx: Index of row to update
            patch: Patch to apply
            run_id: Run ID for audit
            enforce_preconditions: Whether to check preconditions
            
        Returns:
            PatchApplicationResult
        """
        row = df.loc[row_idx]
        column_name = patch.column_name
        
        # Check if column exists
        if column_name not in df.columns:
            logger.warning("Column not found for patch",
                          column=column_name,
                          row_uuid=patch.row_uuid)
            return PatchApplicationResult(
                patch=patch,
                applied=False,
                reason=f"Column '{column_name}' not found"
            )
        
        # Get current value
        current_value = row[column_name]
        current_value_str = str(current_value) if pd.notna(current_value) else None
        
        # Check confidence threshold
        if patch.confidence < settings.confidence_floor:
            logger.debug("Patch below confidence threshold",
                        confidence=patch.confidence,
                        threshold=settings.confidence_floor,
                        row_uuid=patch.row_uuid,
                        column=column_name)
            return PatchApplicationResult(
                patch=patch,
                applied=False,
                reason=f"Confidence {patch.confidence} below threshold {settings.confidence_floor}"
            )
        
        # Check precondition
        if enforce_preconditions and patch.before_value is not None:
            # Normalize for comparison
            before_normalized = self._normalize_value(patch.before_value)
            current_normalized = self._normalize_value(current_value_str)
            
            if before_normalized != current_normalized:
                logger.debug("Precondition failed",
                            expected=patch.before_value,
                            actual=current_value_str,
                            row_uuid=patch.row_uuid,
                            column=column_name)
                return PatchApplicationResult(
                    patch=patch,
                    applied=False,
                    reason=f"Precondition failed: expected '{patch.before_value}', found '{current_value_str}'",
                    actual_value=current_value_str
                )
        
        # Apply the patch
        df.at[row_idx, column_name] = patch.after_value
        
        # Create audit event
        audit = AuditEvent(
            run_id=run_id,
            row_uuid=patch.row_uuid,
            column_name=column_name,
            before_value=current_value_str,
            after_value=patch.after_value,
            source=patch.source,
            rule_id=patch.rule_id,
            contract_id=patch.contract_id,
            reason=patch.reason,
            confidence=patch.confidence
        )
        self.audit_events.append(audit)
        
        # Create diff entry
        diff = DiffEntry(
            row_number=patch.row_number,
            row_uuid=patch.row_uuid,
            column_name=column_name,
            before_value=current_value_str,
            after_value=patch.after_value,
            source=patch.source,
            reason=patch.reason
        )
        self.diff_entries.append(diff)
        
        logger.debug("Patch applied successfully",
                    row_uuid=patch.row_uuid,
                    column=column_name,
                    before=current_value_str,
                    after=patch.after_value,
                    source=patch.source.value)
        
        return PatchApplicationResult(
            patch=patch,
            applied=True,
            reason="Applied successfully"
        )
    
    def _normalize_value(self, value: Optional[str]) -> Optional[str]:
        """
        Normalize value for precondition comparison.
        
        Args:
            value: Value to normalize
            
        Returns:
            Normalized value
        """
        if value is None or value == '':
            return None
        
        # Convert to string and strip whitespace
        normalized = str(value).strip()
        
        # Normalize empty strings to None
        if normalized == '' or normalized.lower() in ['nan', 'none', 'null']:
            return None
        
        return normalized
    
    async def persist_audit_events(self) -> int:
        """
        Persist audit events to database.
        
        Returns:
            Number of events persisted
        """
        if not self.db or not self.audit_events:
            return 0
        
        # Convert to dict format for batch insert
        events_data = []
        for event in self.audit_events:
            events_data.append({
                'run_id': event.run_id,
                'row_uuid': event.row_uuid,
                'column_name': event.column_name,
                'before_value': event.before_value,
                'after_value': event.after_value,
                'source': event.source.value,
                'rule_id': event.rule_id,
                'contract_id': event.contract_id,
                'reason': event.reason,
                'confidence': event.confidence
            })
        
        count = await self.db.create_audit_events_batch(events_data)
        logger.info("Persisted audit events", count=count)
        return count
    
    def generate_audit_report(self) -> Dict[str, Any]:
        """Generate summary report of audit events."""
        report = {
            'total_changes': len(self.audit_events),
            'by_source': {},
            'by_column': {},
            'by_rule': {},
            'confidence_distribution': {
                '1.0': 0,
                '0.9-0.99': 0,
                '0.8-0.89': 0,
                'below_0.8': 0
            }
        }
        
        for event in self.audit_events:
            # By source
            source = event.source.value
            if source not in report['by_source']:
                report['by_source'][source] = 0
            report['by_source'][source] += 1
            
            # By column
            if event.column_name not in report['by_column']:
                report['by_column'][event.column_name] = 0
            report['by_column'][event.column_name] += 1
            
            # By rule
            if event.rule_id:
                if event.rule_id not in report['by_rule']:
                    report['by_rule'][event.rule_id] = 0
                report['by_rule'][event.rule_id] += 1
            
            # Confidence distribution
            if event.confidence is not None:
                if event.confidence == 1.0:
                    report['confidence_distribution']['1.0'] += 1
                elif event.confidence >= 0.9:
                    report['confidence_distribution']['0.9-0.99'] += 1
                elif event.confidence >= 0.8:
                    report['confidence_distribution']['0.8-0.89'] += 1
                else:
                    report['confidence_distribution']['below_0.8'] += 1
        
        return report
    
    def export_diff_to_csv(self, output_path: str) -> int:
        """
        Export diff entries to CSV file.
        
        Args:
            output_path: Path for output CSV
            
        Returns:
            Number of entries exported
        """
        if not self.diff_entries:
            logger.warning("No diff entries to export")
            return 0
        
        # Convert to DataFrame
        diff_data = [entry.to_csv_dict() for entry in self.diff_entries]
        diff_df = pd.DataFrame(diff_data)
        
        # Sort by row number
        diff_df = diff_df.sort_values('row_number')
        
        # Export to CSV
        diff_df.to_csv(output_path, index=False)
        
        logger.info("Exported diff to CSV", 
                   path=output_path,
                   entries=len(diff_df))
        
        return len(diff_df)
    
    def export_audit_to_ndjson(self, output_path: str) -> int:
        """
        Export audit events to NDJSON file.
        
        Args:
            output_path: Path for output NDJSON
            
        Returns:
            Number of events exported
        """
        if not self.audit_events:
            logger.warning("No audit events to export")
            return 0
        
        with open(output_path, 'w') as f:
            for event in self.audit_events:
                json_line = json.dumps(event.to_ndjson_dict())
                f.write(json_line + '\n')
        
        logger.info("Exported audit to NDJSON",
                   path=output_path,
                   events=len(self.audit_events))
        
        return len(self.audit_events)
    
    def get_precondition_failures(self) -> List[PatchApplicationResult]:
        """Get list of patches that failed precondition checks."""
        return [
            result for result in self.application_results
            if not result.applied and 'precondition' in result.reason.lower()
        ]
    
    def get_application_summary(self) -> Dict[str, Any]:
        """Get summary of patch application results."""
        applied = sum(1 for r in self.application_results if r.applied)
        failed = len(self.application_results) - applied
        
        failure_reasons = {}
        for result in self.application_results:
            if not result.applied:
                reason_type = self._categorize_failure_reason(result.reason)
                if reason_type not in failure_reasons:
                    failure_reasons[reason_type] = 0
                failure_reasons[reason_type] += 1
        
        return {
            'total_patches': len(self.application_results),
            'applied': applied,
            'failed': failed,
            'success_rate': applied / len(self.application_results) if self.application_results else 0,
            'failure_reasons': failure_reasons,
            'audit_events_created': len(self.audit_events),
            'diff_entries_created': len(self.diff_entries)
        }
    
    def _categorize_failure_reason(self, reason: str) -> str:
        """Categorize failure reason for reporting."""
        reason_lower = reason.lower()
        if 'precondition' in reason_lower:
            return 'precondition_failed'
        elif 'confidence' in reason_lower:
            return 'low_confidence'
        elif 'not found' in reason_lower:
            return 'not_found'
        elif 'column' in reason_lower:
            return 'column_missing'
        else:
            return 'other'