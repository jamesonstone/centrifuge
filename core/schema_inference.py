"""Experimental schema inference from CSV data."""

import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import structlog

from core.models import (
    Schema, ColumnDefinition, DataType, ColumnPolicy
)

logger = structlog.get_logger()


class SchemaInferencer:
    """Infer schema from CSV data (experimental feature)."""
    
    def __init__(self, sample_size: int = 1000):
        """
        Initialize schema inferencer.
        
        Args:
            sample_size: Number of rows to sample for inference
        """
        self.sample_size = sample_size
        self.date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # ISO date
            r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
            r'^\d{2}-\d{2}-\d{4}$',  # MM-DD-YYYY
            r'^\d{4}/\d{2}/\d{2}$',  # YYYY/MM/DD
        ]
        self.reference_patterns = [
            r'^REF-?\d+$',
            r'^INV-?\d+$',
            r'^TXN-?\d+$',
            r'^[A-Z]{2,4}-?\d+$',
        ]
        
    def infer_schema(
        self,
        df: pd.DataFrame,
        use_experimental: bool = False
    ) -> Schema:
        """
        Infer schema from dataframe.
        
        Args:
            df: Input dataframe
            use_experimental: Enable experimental features
            
        Returns:
            Inferred schema
        """
        if not use_experimental:
            logger.warning("Schema inference called without experimental flag")
            return self._create_default_schema(df)
        
        logger.info("Starting experimental schema inference", 
                   columns=len(df.columns),
                   rows=len(df))
        
        # Sample data for analysis
        sample_df = df.head(self.sample_size) if len(df) > self.sample_size else df
        
        columns = []
        header_aliases = {}
        
        for col_name in df.columns:
            # Infer column definition
            col_def = self._infer_column(col_name, sample_df[col_name])
            columns.append(col_def)
            
            # Create header alias mapping
            canonical_name = self._canonicalize_column_name(col_name)
            if canonical_name != col_name:
                header_aliases[col_name] = canonical_name
        
        # Detect primary key
        self._detect_primary_key(columns, sample_df)
        
        # Apply business rules
        self._apply_business_rules(columns)
        
        schema = Schema(
            version="inferred-1.0.0",
            columns=columns,
            header_aliases=header_aliases,
            metadata={
                "inferred": True,
                "sample_size": len(sample_df),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info("Schema inference complete",
                   columns_inferred=len(columns),
                   aliases_created=len(header_aliases))
        
        return schema
    
    def _infer_column(
        self,
        col_name: str,
        series: pd.Series
    ) -> ColumnDefinition:
        """
        Infer column definition from data.
        
        Args:
            col_name: Column name
            series: Data series
            
        Returns:
            Inferred column definition
        """
        canonical_name = self._canonicalize_column_name(col_name)
        
        # Check if mostly null
        non_null = series.dropna()
        if len(non_null) == 0:
            return ColumnDefinition(
                name=canonical_name,
                display_name=col_name,
                data_type=DataType.STRING,
                required=False,
                policy=ColumnPolicy.RULE_ONLY
            )
        
        # Determine data type
        data_type = self._infer_data_type(non_null)
        
        # Check if required (has non-null values in most rows)
        required = (len(non_null) / len(series)) > 0.95
        
        # Determine policy based on column characteristics
        policy = self._infer_policy(canonical_name, non_null, data_type)
        
        # Check for enums (limited unique values)
        allowed_values = None
        unique_ratio = len(non_null.unique()) / len(non_null)
        if data_type == DataType.STRING and unique_ratio < 0.1:
            # Likely an enum
            allowed_values = sorted(non_null.unique().tolist())
        
        # Check for patterns
        pattern = self._infer_pattern(canonical_name, non_null)
        
        return ColumnDefinition(
            name=canonical_name,
            display_name=col_name,
            data_type=data_type,
            required=required,
            policy=policy,
            allowed_values=allowed_values[:20] if allowed_values else None,  # Limit to 20
            pattern=pattern,
            allow_in_prompt=policy == ColumnPolicy.LLM_ALLOWED
        )
    
    def _infer_data_type(self, series: pd.Series) -> DataType:
        """
        Infer data type from series.
        
        Args:
            series: Non-null data series
            
        Returns:
            Inferred data type
        """
        # Sample values
        samples = series.head(100).tolist()
        
        # Try to parse as different types
        type_counts = {
            DataType.INTEGER: 0,
            DataType.DECIMAL: 0,
            DataType.DATE: 0,
            DataType.BOOLEAN: 0,
            DataType.STRING: 0
        }
        
        for value in samples:
            value_str = str(value).strip()
            
            # Check boolean
            if value_str.lower() in ['true', 'false', 'yes', 'no', '1', '0']:
                type_counts[DataType.BOOLEAN] += 1
                continue
            
            # Check integer
            try:
                int(value_str)
                type_counts[DataType.INTEGER] += 1
                continue
            except ValueError:
                pass
            
            # Check decimal
            try:
                float(value_str.replace(',', '').replace('$', ''))
                type_counts[DataType.DECIMAL] += 1
                continue
            except ValueError:
                pass
            
            # Check date
            if self._is_date(value_str):
                type_counts[DataType.DATE] += 1
                continue
            
            # Default to string
            type_counts[DataType.STRING] += 1
        
        # Return most common type
        return max(type_counts, key=type_counts.get)
    
    def _is_date(self, value: str) -> bool:
        """Check if value matches date pattern."""
        for pattern in self.date_patterns:
            if re.match(pattern, value):
                return True
        
        # Try parsing common date formats
        date_formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y',
            '%Y/%m/%d', '%B %d, %Y', '%d %B %Y'
        ]
        
        for fmt in date_formats:
            try:
                datetime.strptime(value, fmt)
                return True
            except ValueError:
                continue
        
        return False
    
    def _infer_policy(
        self,
        col_name: str,
        series: pd.Series,
        data_type: DataType
    ) -> ColumnPolicy:
        """
        Infer column policy.
        
        Args:
            col_name: Canonical column name
            series: Data series
            data_type: Inferred data type
            
        Returns:
            Column policy
        """
        # LLM-allowed columns (experimental)
        llm_candidates = [
            'department', 'category', 'type', 'status',
            'account_name', 'account', 'vendor', 'customer'
        ]
        
        if col_name.lower() in llm_candidates and data_type == DataType.STRING:
            # Check if it has reasonable variety for LLM
            unique_count = len(series.unique())
            if 5 <= unique_count <= 100:
                return ColumnPolicy.LLM_ALLOWED
        
        # Default to rule-only
        return ColumnPolicy.RULE_ONLY
    
    def _infer_pattern(
        self,
        col_name: str,
        series: pd.Series
    ) -> Optional[str]:
        """
        Infer regex pattern for column.
        
        Args:
            col_name: Column name
            series: Data series
            
        Returns:
            Regex pattern or None
        """
        # Check for reference number patterns
        if 'ref' in col_name.lower() or 'number' in col_name.lower():
            # Check if values match common patterns
            for pattern in self.reference_patterns:
                matches = series.apply(lambda x: bool(re.match(pattern, str(x), re.IGNORECASE)))
                if matches.mean() > 0.9:  # 90% match
                    return pattern
        
        # Check for ID patterns
        if 'id' in col_name.lower():
            # Check for UUID pattern
            uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
            matches = series.apply(lambda x: bool(re.match(uuid_pattern, str(x), re.IGNORECASE)))
            if matches.mean() > 0.9:
                return uuid_pattern
        
        return None
    
    def _canonicalize_column_name(self, name: str) -> str:
        """
        Convert column name to canonical form.
        
        Args:
            name: Original column name
            
        Returns:
            Canonical name (snake_case)
        """
        # Remove special characters
        name = re.sub(r'[^\w\s]', '', name)
        
        # Convert to snake_case
        name = re.sub(r'\s+', '_', name)
        name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
        name = name.lower()
        
        # Remove duplicate underscores
        name = re.sub(r'_+', '_', name)
        name = name.strip('_')
        
        return name
    
    def _detect_primary_key(
        self,
        columns: List[ColumnDefinition],
        df: pd.DataFrame
    ) -> None:
        """
        Detect and mark primary key column.
        
        Args:
            columns: List of column definitions
            df: Sample dataframe
        """
        for col_def in columns:
            col_data = df[col_def.display_name]
            
            # Check if column could be primary key
            if col_def.data_type in [DataType.STRING, DataType.INTEGER]:
                # Check uniqueness
                if col_data.nunique() == len(col_data):
                    # Check if it looks like an ID
                    if 'id' in col_def.name.lower() or \
                       'key' in col_def.name.lower() or \
                       'number' in col_def.name.lower():
                        col_def.is_primary_key = True
                        logger.info("Primary key detected", column=col_def.name)
                        break
    
    def _apply_business_rules(self, columns: List[ColumnDefinition]) -> None:
        """
        Apply standard business rules to columns.
        
        Args:
            columns: List of column definitions
        """
        # Find debit/credit columns
        debit_col = None
        credit_col = None
        
        for col in columns:
            if 'debit' in col.name.lower():
                debit_col = col
            elif 'credit' in col.name.lower():
                credit_col = col
        
        # Apply debit/credit rules
        if debit_col and credit_col:
            # These should be decimal and not required individually
            debit_col.data_type = DataType.DECIMAL
            credit_col.data_type = DataType.DECIMAL
            debit_col.required = False
            credit_col.required = False
            
            logger.info("Applied debit/credit business rules")
    
    def _create_default_schema(self, df: pd.DataFrame) -> Schema:
        """
        Create minimal default schema without inference.
        
        Args:
            df: Input dataframe
            
        Returns:
            Default schema
        """
        columns = []
        for col_name in df.columns:
            canonical = self._canonicalize_column_name(col_name)
            columns.append(
                ColumnDefinition(
                    name=canonical,
                    display_name=col_name,
                    data_type=DataType.STRING,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                )
            )
        
        return Schema(
            version="default-1.0.0",
            columns=columns
        )


class ExperimentalFeatures:
    """Manager for experimental features."""
    
    def __init__(self):
        """Initialize experimental features manager."""
        self.features = {
            'use_inferred': False,
            'allow_in_prompt': {},
            'extended_llm_columns': [],
            'auto_quarantine': True,
            'parallel_processing': False
        }
        
    def enable_schema_inference(self) -> None:
        """Enable experimental schema inference."""
        logger.warning("EXPERIMENTAL: Schema inference enabled")
        self.features['use_inferred'] = True
        
    def set_llm_columns(self, columns: List[str]) -> None:
        """
        Set which columns can use LLM.
        
        Args:
            columns: List of column names
        """
        logger.warning(f"EXPERIMENTAL: LLM enabled for columns: {columns}")
        self.features['extended_llm_columns'] = columns
        
    def set_column_prompt_permission(
        self,
        column: str,
        allow: bool
    ) -> None:
        """
        Set whether column can be included in prompts.
        
        Args:
            column: Column name
            allow: Whether to allow in prompts
        """
        self.features['allow_in_prompt'][column] = allow
        if allow:
            logger.warning(f"EXPERIMENTAL: Column '{column}' allowed in prompts")
        
    def is_experimental_enabled(self, feature: str) -> bool:
        """
        Check if experimental feature is enabled.
        
        Args:
            feature: Feature name
            
        Returns:
            Whether feature is enabled
        """
        return self.features.get(feature, False)
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current experimental configuration."""
        return self.features.copy()
    
    def validate_experimental_request(
        self,
        options: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate experimental feature request.
        
        Args:
            options: Request options
            
        Returns:
            Tuple of (valid, error_message)
        """
        if options.get('use_inferred', False):
            if not options.get('acknowledge_experimental', False):
                return False, "Must acknowledge experimental features with 'acknowledge_experimental=true'"
            
        if options.get('llm_columns'):
            # Check if columns are in allowed list
            allowed = ['department', 'account_name', 'category', 'type', 'status']
            for col in options['llm_columns']:
                if col not in allowed and not options.get('force_llm_columns', False):
                    return False, f"Column '{col}' not in allowed LLM columns. Use 'force_llm_columns=true' to override."
        
        return True, None