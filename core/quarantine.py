"""Post-processing validation and quarantine management."""

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import pandas as pd
import structlog

from core.models import (
    Schema, ErrorCategory, QuarantineRow,
    Patch, AuditEvent
)
from core.validation import Validator, ValidationError
from core.config import settings

logger = structlog.get_logger()


class QuarantineManager:
    """Manages quarantine process for rows that fail final validation."""
    
    def __init__(self, schema: Schema):
        """
        Initialize quarantine manager.
        
        Args:
            schema: Schema for validation
        """
        self.schema = schema
        self.validator = Validator(schema)
        self.quarantine_rows: List[QuarantineRow] = []
        self.validation_errors: List[ValidationError] = []
        self.quarantine_stats: Dict[str, Any] = {}
        
    def validate_and_quarantine(
        self,
        df: pd.DataFrame,
        attempted_patches: List[Patch] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[QuarantineRow], Dict[str, Any]]:
        """
        Re-validate data and quarantine rows that still fail.
        
        Args:
            df: DataFrame to validate (after all processing)
            attempted_patches: Patches that were attempted but may have failed
            
        Returns:
            Tuple of (clean_df, quarantine_df, quarantine_rows, statistics)
        """
        logger.info("Starting post-processing validation", rows=len(df))
        
        # Reset state
        self.quarantine_rows = []
        self.validation_errors = []
        
        # Run validation
        valid_df, quarantine_df, errors = self.validator.validate(df)
        self.validation_errors = errors
        
        # Build patch lookup for attempted fixes
        patches_by_row: Dict[UUID, List[Patch]] = defaultdict(list)
        if attempted_patches:
            for patch in attempted_patches:
                patches_by_row[patch.row_uuid].append(patch)
        
        # Process quarantined rows
        if len(quarantine_df) > 0:
            for idx, row in quarantine_df.iterrows():
                row_uuid = UUID(row['row_uuid'])
                row_number = row['row_number']
                
                # Get errors for this row
                row_errors = [
                    e for e in self.validation_errors 
                    if e.row_number == row_number
                ]
                
                # Categorize the primary error
                category = self._categorize_errors(row_errors)
                
                # Get error details
                error_details = [e.message for e in row_errors]
                
                # Get attempted fixes for this row
                attempted_fixes = patches_by_row.get(row_uuid, [])
                
                # Create quarantine row
                quarantine_row = QuarantineRow(
                    row_uuid=row_uuid,
                    row_number=row_number,
                    error_category=category,
                    error_details=error_details,
                    original_data=row.to_dict(),
                    attempted_fixes=attempted_fixes
                )
                self.quarantine_rows.append(quarantine_row)
        
        # Calculate statistics
        self.quarantine_stats = self._calculate_statistics(
            len(df), len(valid_df), len(quarantine_df)
        )
        
        logger.info("Quarantine complete",
                   clean_rows=len(valid_df),
                   quarantined_rows=len(quarantine_df),
                   error_count=len(self.validation_errors))
        
        return valid_df, quarantine_df, self.quarantine_rows, self.quarantine_stats
    
    def _categorize_errors(self, errors: List[ValidationError]) -> ErrorCategory:
        """
        Determine primary error category for a set of errors.
        
        Args:
            errors: List of validation errors for a row
            
        Returns:
            Primary ErrorCategory
        """
        if not errors:
            return ErrorCategory.VALIDATION_FAILURE
        
        # Priority order for categorization
        for error in errors:
            # Check for specific error types
            if error.error_type == 'debit_xor_credit':
                return ErrorCategory.VALIDATION_FAILURE
            elif error.error_type in ['missing_column', 'required_field']:
                return ErrorCategory.VALIDATION_FAILURE
            elif error.error_type == 'type_mismatch':
                return ErrorCategory.PARSE_ERROR
            elif error.error_type in ['missing_primary_key', 'duplicate_primary_key']:
                return ErrorCategory.VALIDATION_FAILURE
            elif 'llm' in error.error_type.lower():
                return ErrorCategory.LLM_CONTRACT_FAILURE
            elif 'confidence' in error.error_type.lower():
                return ErrorCategory.LOW_CONFIDENCE
            elif 'edit_cap' in error.error_type.lower():
                return ErrorCategory.EDIT_CAP_EXCEEDED
        
        # Default category
        return ErrorCategory.VALIDATION_FAILURE
    
    def _calculate_statistics(
        self,
        total_rows: int,
        clean_rows: int,
        quarantined_rows: int
    ) -> Dict[str, Any]:
        """
        Calculate quarantine statistics.
        
        Args:
            total_rows: Total input rows
            clean_rows: Successfully cleaned rows
            quarantined_rows: Quarantined rows
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_rows': total_rows,
            'clean_rows': clean_rows,
            'quarantined_rows': quarantined_rows,
            'success_rate': clean_rows / total_rows if total_rows > 0 else 0,
            'quarantine_rate': quarantined_rows / total_rows if total_rows > 0 else 0,
            'errors_by_category': defaultdict(int),
            'errors_by_type': defaultdict(int),
            'errors_by_column': defaultdict(int),
            'rows_with_attempted_fixes': 0,
            'top_error_messages': []
        }
        
        # Count errors by category
        for qrow in self.quarantine_rows:
            stats['errors_by_category'][qrow.error_category.value] += 1
            if qrow.attempted_fixes:
                stats['rows_with_attempted_fixes'] += 1
        
        # Count errors by type and column
        for error in self.validation_errors:
            stats['errors_by_type'][error.error_type] += 1
            stats['errors_by_column'][error.column_name] += 1
        
        # Get top error messages
        error_message_counts = defaultdict(int)
        for error in self.validation_errors:
            error_message_counts[error.message] += 1
        
        # Sort and get top 10
        top_messages = sorted(
            error_message_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        stats['top_error_messages'] = [
            {'message': msg, 'count': count}
            for msg, count in top_messages
        ]
        
        return stats
    
    def export_errors_csv(self, output_path: str) -> int:
        """
        Export quarantined rows to errors.csv.
        
        Args:
            output_path: Path for output CSV
            
        Returns:
            Number of rows exported
        """
        if not self.quarantine_rows:
            logger.warning("No quarantine rows to export")
            return 0
        
        # Convert to list of dicts for DataFrame
        error_data = []
        for qrow in self.quarantine_rows:
            # Start with metadata
            row_dict = {
                'row_number': qrow.row_number,
                'error_category': qrow.error_category.value,
                'error_count': len(qrow.error_details),
                'error_summary': '; '.join(qrow.error_details[:3])  # First 3 errors
            }
            
            # Add original data columns (excluding internal fields)
            for key, value in qrow.original_data.items():
                if key not in ['row_uuid', 'row_number']:
                    row_dict[key] = value
            
            # Add attempted fixes summary
            if qrow.attempted_fixes:
                fixes_summary = f"{len(qrow.attempted_fixes)} fixes attempted"
                row_dict['attempted_fixes'] = fixes_summary
            else:
                row_dict['attempted_fixes'] = 'None'
            
            error_data.append(row_dict)
        
        # Create DataFrame
        errors_df = pd.DataFrame(error_data)
        
        # Reorder columns for better readability
        metadata_cols = ['row_number', 'error_category', 'error_count', 'error_summary', 'attempted_fixes']
        data_cols = [col for col in errors_df.columns if col not in metadata_cols]
        ordered_cols = metadata_cols + data_cols
        errors_df = errors_df[ordered_cols]
        
        # Sort by row number
        errors_df = errors_df.sort_values('row_number')
        
        # Export to CSV
        errors_df.to_csv(output_path, index=False)
        
        logger.info("Exported errors to CSV",
                   path=output_path,
                   rows=len(errors_df))
        
        return len(errors_df)
    
    def export_detailed_errors_json(self, output_path: str) -> int:
        """
        Export detailed error information to JSON.
        
        Args:
            output_path: Path for output JSON
            
        Returns:
            Number of rows exported
        """
        if not self.quarantine_rows:
            logger.warning("No quarantine rows to export")
            return 0
        
        detailed_errors = []
        
        for qrow in self.quarantine_rows:
            error_detail = {
                'row_uuid': str(qrow.row_uuid),
                'row_number': qrow.row_number,
                'error_category': qrow.error_category.value,
                'error_details': qrow.error_details,
                'original_data': qrow.original_data,
                'attempted_fixes': []
            }
            
            # Add attempted fixes with details
            for patch in qrow.attempted_fixes:
                fix_detail = {
                    'column': patch.column_name,
                    'before_value': patch.before_value,
                    'after_value': patch.after_value,
                    'source': patch.source.value,
                    'confidence': patch.confidence,
                    'reason': patch.reason
                }
                error_detail['attempted_fixes'].append(fix_detail)
            
            # Get specific validation errors for this row
            row_validation_errors = [
                {
                    'column': e.column_name,
                    'value': str(e.value) if e.value is not None else None,
                    'error_type': e.error_type,
                    'message': e.message
                }
                for e in self.validation_errors
                if e.row_number == qrow.row_number
            ]
            error_detail['validation_errors'] = row_validation_errors
            
            detailed_errors.append(error_detail)
        
        # Write to JSON
        with open(output_path, 'w') as f:
            json.dump(detailed_errors, f, indent=2, default=str)
        
        logger.info("Exported detailed errors to JSON",
                   path=output_path,
                   rows=len(detailed_errors))
        
        return len(detailed_errors)
    
    def get_quarantine_summary(self) -> Dict[str, Any]:
        """Get summary of quarantine results."""
        summary = {
            'total_quarantined': len(self.quarantine_rows),
            'total_errors': len(self.validation_errors),
            'statistics': self.quarantine_stats,
            'category_breakdown': {},
            'common_issues': [],
            'fixable_rows': 0,
            'unfixable_rows': 0
        }
        
        # Category breakdown
        for qrow in self.quarantine_rows:
            category = qrow.error_category.value
            if category not in summary['category_breakdown']:
                summary['category_breakdown'][category] = {
                    'count': 0,
                    'percentage': 0,
                    'examples': []
                }
            summary['category_breakdown'][category]['count'] += 1
        
        # Calculate percentages
        total = len(self.quarantine_rows)
        for category_data in summary['category_breakdown'].values():
            category_data['percentage'] = (
                category_data['count'] / total * 100 if total > 0 else 0
            )
        
        # Identify common issues
        issue_patterns = defaultdict(int)
        for error in self.validation_errors:
            pattern = f"{error.column_name}:{error.error_type}"
            issue_patterns[pattern] += 1
        
        # Get top 5 issues
        top_issues = sorted(
            issue_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        summary['common_issues'] = [
            {'pattern': pattern, 'occurrences': count}
            for pattern, count in top_issues
        ]
        
        # Categorize fixable vs unfixable
        for qrow in self.quarantine_rows:
            # Consider rows with certain error types as potentially fixable
            if qrow.error_category in [
                ErrorCategory.LOW_CONFIDENCE,
                ErrorCategory.EDIT_CAP_EXCEEDED
            ]:
                summary['fixable_rows'] += 1
            else:
                summary['unfixable_rows'] += 1
        
        return summary
    
    def generate_remediation_report(self) -> List[Dict[str, Any]]:
        """
        Generate recommendations for fixing quarantined rows.
        
        Returns:
            List of remediation recommendations
        """
        recommendations = []
        
        # Analyze patterns
        patterns = {
            'missing_required': [],
            'type_errors': [],
            'business_rule_violations': [],
            'low_confidence_mappings': [],
            'duplicate_keys': []
        }
        
        for qrow in self.quarantine_rows:
            for error_detail in qrow.error_details:
                if 'required' in error_detail.lower():
                    patterns['missing_required'].append(qrow.row_number)
                elif 'type' in error_detail.lower():
                    patterns['type_errors'].append(qrow.row_number)
                elif 'debit' in error_detail.lower() or 'credit' in error_detail.lower():
                    patterns['business_rule_violations'].append(qrow.row_number)
                elif 'confidence' in error_detail.lower():
                    patterns['low_confidence_mappings'].append(qrow.row_number)
                elif 'duplicate' in error_detail.lower():
                    patterns['duplicate_keys'].append(qrow.row_number)
        
        # Generate recommendations
        if patterns['missing_required']:
            recommendations.append({
                'issue': 'Missing Required Fields',
                'affected_rows': len(set(patterns['missing_required'])),
                'recommendation': 'Review source data for completeness. Consider data enrichment or default values.',
                'priority': 'High',
                'example_rows': list(set(patterns['missing_required']))[:5]
            })
        
        if patterns['type_errors']:
            recommendations.append({
                'issue': 'Data Type Mismatches',
                'affected_rows': len(set(patterns['type_errors'])),
                'recommendation': 'Implement additional data cleaning rules or adjust schema expectations.',
                'priority': 'Medium',
                'example_rows': list(set(patterns['type_errors']))[:5]
            })
        
        if patterns['business_rule_violations']:
            recommendations.append({
                'issue': 'Business Rule Violations',
                'affected_rows': len(set(patterns['business_rule_violations'])),
                'recommendation': 'Review debit/credit entries for accounting accuracy. May require manual correction.',
                'priority': 'High',
                'example_rows': list(set(patterns['business_rule_violations']))[:5]
            })
        
        if patterns['low_confidence_mappings']:
            recommendations.append({
                'issue': 'Low Confidence Mappings',
                'affected_rows': len(set(patterns['low_confidence_mappings'])),
                'recommendation': 'Review and manually verify mappings. Consider adding to canonical lists.',
                'priority': 'Low',
                'example_rows': list(set(patterns['low_confidence_mappings']))[:5]
            })
        
        if patterns['duplicate_keys']:
            recommendations.append({
                'issue': 'Duplicate Transaction IDs',
                'affected_rows': len(set(patterns['duplicate_keys'])),
                'recommendation': 'Investigate duplicate entries. May indicate data quality issues at source.',
                'priority': 'High',
                'example_rows': list(set(patterns['duplicate_keys']))[:5]
            })
        
        return recommendations