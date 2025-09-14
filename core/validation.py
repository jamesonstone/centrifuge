"""Data validation module for Centrifuge."""

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

import pandas as pd
import structlog
from uuid_extensions import uuid7
from dateutil import parser as date_parser

from core.models import (
    Schema, ColumnType, ErrorCategory,
    QuarantineRow, Patch, SourceType
)
from core.config import settings

logger = structlog.get_logger()


class ValidationError:
    """Represents a validation error for a specific cell or row."""

    def __init__(
        self,
        row_uuid: UUID,
        row_number: int,
        column_name: str,
        value: Any,
        error_type: str,
        message: str,
        category: ErrorCategory = ErrorCategory.VALIDATION_FAILURE
    ):
        self.row_uuid = row_uuid
        self.row_number = row_number
        self.column_name = column_name
        self.value = value
        self.error_type = error_type
        self.message = message
        self.category = category

    def __str__(self):
        return f"Row {self.row_number}, {self.column_name}: {self.message}"


class Validator:
    """Validates DataFrame against schema and business rules."""

    # Common date formats to try
    DATE_FORMATS = [
        '%Y-%m-%d',           # 2024-01-15
        '%m/%d/%Y',           # 01/16/2024
        '%d/%m/%Y',           # 16/01/2024
        '%Y/%m/%d',           # 2024/01/15
        '%Y.%m.%d',           # 2024.01.15
        '%d-%m-%Y',           # 15-01-2024
        '%m-%d-%Y',           # 01-15-2024
        '%d.%m.%Y',           # 15.01.2024
        '%b %d %Y',           # Jan 15 2024
        '%B %d %Y',           # January 15 2024
        '%d %b %Y',           # 15 Jan 2024
        '%d %B %Y',           # 15 January 2024
        '%Y-%b-%d',           # 2024-Jan-15
        '%d-%b-%Y',           # 15-Jan-2024
        '%m/%d/%y',           # 01/16/24
        '%d/%m/%y',           # 16/01/24
        '%y/%m/%d',           # 24/01/16
    ]

    def __init__(self, schema: Schema):
        """
        Initialize validator with schema.

        Args:
            schema: Schema defining validation rules
        """
        self.schema = schema
        self.validation_errors: List[ValidationError] = []
        self.quarantine_rows: List[QuarantineRow] = []

    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[ValidationError]]:
        """
        Validate DataFrame against schema and business rules.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (valid_df, quarantine_df, errors)
        """
        logger.info("Starting validation", rows=len(df))

        # Reset error tracking
        self.validation_errors = []
        self.quarantine_rows = []

        # Create a copy to work with
        working_df = df.copy()

        # Track which rows to quarantine
        rows_to_quarantine = set()

        # Phase 1: Check required columns
        missing_columns = self._check_required_columns(working_df)
        if missing_columns:
            logger.error("Missing required columns", columns=missing_columns)
            # Can't proceed without required columns
            for col in missing_columns:
                for idx, row in working_df.iterrows():
                    self.validation_errors.append(
                        ValidationError(
                            row_uuid=UUID(row['row_uuid']),
                            row_number=row['row_number'],
                            column_name=col,
                            value=None,
                            error_type='missing_column',
                            message=f"Required column '{col}' is missing"
                        )
                    )
                    rows_to_quarantine.add(idx)

        # Phase 2: Validate column types and constraints
        for col_name, col_def in self.schema.columns.items():
            if col_name not in working_df.columns:
                continue

            for idx, row in working_df.iterrows():
                value = row[col_name]
                row_uuid = UUID(row['row_uuid'])
                row_number = row['row_number']

                # Check required fields
                if col_def.required and (pd.isna(value) or value == '' or value is None):
                    self.validation_errors.append(
                        ValidationError(
                            row_uuid=row_uuid,
                            row_number=row_number,
                            column_name=col_name,
                            value=value,
                            error_type='required_field',
                            message=f"Required field '{col_name}' is empty"
                        )
                    )
                    rows_to_quarantine.add(idx)
                    continue

                # Skip validation for null values in optional fields
                if pd.isna(value):
                    continue

                # Validate by type
                is_valid = self._validate_type(value, col_def.type, col_name)
                if not is_valid:
                    self.validation_errors.append(
                        ValidationError(
                            row_uuid=row_uuid,
                            row_number=row_number,
                            column_name=col_name,
                            value=value,
                            error_type='type_mismatch',
                            message=f"Value '{value}' is not a valid {col_def.type.value}"
                        )
                    )
                    rows_to_quarantine.add(idx)

                # Check enum constraints
                if col_def.enum_values and value not in col_def.enum_values:
                    # Will be handled by LLM for specific columns
                    if col_name not in ['Department', 'Account Name']:
                        self.validation_errors.append(
                            ValidationError(
                                row_uuid=row_uuid,
                                row_number=row_number,
                                column_name=col_name,
                                value=value,
                                error_type='enum_violation',
                                message=f"Value '{value}' not in allowed values"
                            )
                        )
                        rows_to_quarantine.add(idx)

                # Check pattern constraints
                if col_def.pattern and not re.match(col_def.pattern, str(value)):
                    self.validation_errors.append(
                        ValidationError(
                            row_uuid=row_uuid,
                            row_number=row_number,
                            column_name=col_name,
                            value=value,
                            error_type='pattern_violation',
                            message=f"Value '{value}' doesn't match pattern {col_def.pattern}"
                        )
                    )
                    rows_to_quarantine.add(idx)

                # Check length constraints
                if col_def.min_length and len(str(value)) < col_def.min_length:
                    self.validation_errors.append(
                        ValidationError(
                            row_uuid=row_uuid,
                            row_number=row_number,
                            column_name=col_name,
                            value=value,
                            error_type='length_violation',
                            message=f"Value too short (min: {col_def.min_length})"
                        )
                    )
                    rows_to_quarantine.add(idx)

                if col_def.max_length and len(str(value)) > col_def.max_length:
                    self.validation_errors.append(
                        ValidationError(
                            row_uuid=row_uuid,
                            row_number=row_number,
                            column_name=col_name,
                            value=value,
                            error_type='length_violation',
                            message=f"Value too long (max: {col_def.max_length})"
                        )
                    )
                    rows_to_quarantine.add(idx)

        # Phase 3: Check business rules

        # Check debit_xor_credit rule
        if self.schema.constraints.debit_xor_credit:
            # Find debit and credit columns in the schema
            debit_col = next((col for col in self.schema.columns.keys() if 'debit' in col.lower()), None)
            credit_col = next((col for col in self.schema.columns.keys() if 'credit' in col.lower()), None)

            if debit_col and credit_col:
                for idx, row in working_df.iterrows():
                    debit = row.get(debit_col)
                    credit = row.get(credit_col)

                    if not self._validate_debit_xor_credit(debit, credit):
                        self.validation_errors.append(
                            ValidationError(
                                row_uuid=UUID(row['row_uuid']),
                                row_number=row['row_number'],
                                column_name=f'{debit_col}/{credit_col}',
                                value=f"Debit={debit}, Credit={credit}",
                                error_type='debit_xor_credit',
                                message="Exactly one of Debit or Credit must be positive"
                            )
                        )
                        rows_to_quarantine.add(idx)

        # Check Transaction ID format, sentinels, uniqueness (if majority have it)
        if 'Transaction ID' in working_df.columns:
            txn_id_check = self._validate_transaction_ids(working_df)

            # Format validation errors
            for idx in txn_id_check['format_invalid']:
                row = working_df.loc[idx]
                self.validation_errors.append(
                    ValidationError(
                        row_uuid=UUID(row['row_uuid']),
                        row_number=row['row_number'],
                        column_name='Transaction ID',
                        value=row['Transaction ID'],
                        error_type='transaction_id_format',
                        message=f"Invalid Transaction ID format: '{row['Transaction ID']}'. Must be TXN-<digits>",
                        category=ErrorCategory.VALIDATION_FAILURE
                    )
                )
                rows_to_quarantine.add(idx)

            # Missing Transaction ID errors
            for idx in txn_id_check['missing']:
                row = working_df.loc[idx]
                self.validation_errors.append(
                    ValidationError(
                        row_uuid=UUID(row['row_uuid']),
                        row_number=row['row_number'],
                        column_name='Transaction ID',
                        value=None,
                        error_type='missing_primary_key',
                        message="Transaction ID is missing (required when majority have it)"
                    )
                )
                rows_to_quarantine.add(idx)

            # Duplicate Transaction ID errors
            for idx in txn_id_check['duplicates']:
                row = working_df.loc[idx]
                self.validation_errors.append(
                    ValidationError(
                        row_uuid=UUID(row['row_uuid']),
                        row_number=row['row_number'],
                        column_name='Transaction ID',
                        value=row['Transaction ID'],
                        error_type='duplicate_primary_key',
                        message=f"Duplicate Transaction ID: {row['Transaction ID']}"
                    )
                )
                rows_to_quarantine.add(idx)

        # Split into valid and quarantine DataFrames
        if rows_to_quarantine:
            quarantine_df = working_df.loc[list(rows_to_quarantine)]
            valid_df = working_df.drop(list(rows_to_quarantine))

            # Create QuarantineRow objects
            for idx in rows_to_quarantine:
                row = working_df.loc[idx]
                row_errors = [e for e in self.validation_errors
                             if e.row_number == row['row_number']]

                # Determine primary error category
                if any(e.error_type == 'debit_xor_credit' for e in row_errors):
                    category = ErrorCategory.VALIDATION_FAILURE
                elif any(e.error_type in ['missing_column', 'required_field'] for e in row_errors):
                    category = ErrorCategory.VALIDATION_FAILURE
                elif any(e.error_type == 'type_mismatch' for e in row_errors):
                    category = ErrorCategory.PARSE_ERROR
                else:
                    category = ErrorCategory.VALIDATION_FAILURE

                quarantine_row = QuarantineRow(
                    row_uuid=UUID(row['row_uuid']),
                    row_number=row['row_number'],
                    error_category=category,
                    error_details=[e.message for e in row_errors],
                    original_data=row.to_dict()
                )
                self.quarantine_rows.append(quarantine_row)
        else:
            valid_df = working_df
            quarantine_df = pd.DataFrame()

        logger.info("Validation complete",
                   valid_rows=len(valid_df),
                   quarantined_rows=len(quarantine_df),
                   total_errors=len(self.validation_errors))

        return valid_df, quarantine_df, self.validation_errors

    def _check_required_columns(self, df: pd.DataFrame) -> List[str]:
        """Check if all required columns are present."""
        missing = []
        for col_name, col_def in self.schema.columns.items():
            if col_def.required and col_name not in df.columns:
                missing.append(col_name)
        return missing

    def _validate_type(self, value: Any, column_type: ColumnType, column_name: str) -> bool:
        """
        Validate value against column type.

        Args:
            value: Value to validate
            column_type: Expected type
            column_name: Column name for context

        Returns:
            True if valid, False otherwise
        """
        if pd.isna(value):
            return True

        value_str = str(value).strip()

        if column_type == ColumnType.STRING:
            return True  # Everything can be a string

        elif column_type == ColumnType.INTEGER:
            try:
                # Remove commas and currency symbols
                clean_value = value_str.replace(',', '').replace('$', '').strip()
                int(clean_value)
                return True
            except (ValueError, AttributeError):
                return False

        elif column_type == ColumnType.DECIMAL:
            try:
                # Check for infinity or NaN first
                if isinstance(value, float):
                    import math
                    if math.isinf(value) or math.isnan(value):
                        return False

                # Remove commas, quotes, and currency symbols
                clean_value = value_str.replace(',', '').replace('$', '').replace('"', '').strip()

                # Check for string representations of infinity/NaN
                if clean_value.lower() in ['inf', '-inf', 'infinity', '-infinity', 'nan']:
                    return False

                # Handle negative values in parentheses
                if clean_value.startswith('(') and clean_value.endswith(')'):
                    clean_value = '-' + clean_value[1:-1]

                # Try to parse as Decimal and check bounds
                decimal_value = Decimal(clean_value)

                # Check for values that are too large (near float max)
                if abs(decimal_value) > Decimal('1e308'):
                    return False

                return True
            except (InvalidOperation, ValueError, AttributeError):
                return False

        elif column_type == ColumnType.DATE:
            # Try to parse date with various formats
            return self._is_valid_date(value_str)

        elif column_type == ColumnType.BOOLEAN:
            lower_value = value_str.lower()
            return lower_value in ['true', 'false', '1', '0', 'yes', 'no', 'y', 'n']

        return False

    def _is_valid_date(self, value: str) -> bool:
        """Check if value is a valid date."""
        if not value or value.lower() in ['null', 'na', 'n/a']:
            return False

        # Try specific formats first
        for date_format in self.DATE_FORMATS:
            try:
                datetime.strptime(value, date_format)
                return True
            except ValueError:
                continue

        # Try dateutil parser as fallback
        try:
            date_parser.parse(value, fuzzy=False)
            return True
        except (ValueError, TypeError):
            pass

        return False

    def _validate_debit_xor_credit(self, debit: Any, credit: Any) -> bool:
        """
        Validate debit XOR credit business rule.

        Exactly one of debit or credit should be positive, the other should be null/zero.

        Args:
            debit: Debit amount value
            credit: Credit amount value

        Returns:
            True if valid, False otherwise
        """
        # Clean and parse amounts
        debit_amount = self._parse_amount(debit)
        credit_amount = self._parse_amount(credit)

        # Check XOR condition: exactly one should be positive
        debit_positive = debit_amount is not None and debit_amount > 0
        credit_positive = credit_amount is not None and credit_amount > 0

        # Valid if exactly one is positive
        return debit_positive != credit_positive  # XOR

    def _parse_amount(self, value: Any) -> Optional[Decimal]:
        """
        Parse an amount value to Decimal.

        Args:
            value: Amount value (could be string with formatting)

        Returns:
            Decimal value or None if invalid/empty
        """
        if pd.isna(value) or value == '' or value is None:
            return None

        try:
            # Convert to string and clean
            value_str = str(value).strip()

            # Remove quotes
            value_str = value_str.replace('"', '').replace("'", '')

            # Remove currency symbols
            value_str = value_str.replace('$', '').replace('£', '').replace('€', '')

            # Remove commas (thousand separators)
            value_str = value_str.replace(',', '')

            # Handle parentheses for negative values
            if value_str.startswith('(') and value_str.endswith(')'):
                value_str = '-' + value_str[1:-1]

            # Handle empty string after cleaning
            if not value_str or value_str == '-':
                return None

            return Decimal(value_str)
        except (InvalidOperation, ValueError, AttributeError):
            return None

    def _validate_transaction_ids(self, df: pd.DataFrame) -> Dict[str, Set[int]]:
        """
        Validate Transaction ID constraints including format, sentinel tokens, and uniqueness.

        If majority of rows have Transaction ID, enforce uniqueness.

        Args:
            df: DataFrame to validate

        Returns:
            Dict with 'missing', 'duplicates', and 'format_invalid' sets of row indices
        """
        result = {'missing': set(), 'duplicates': set(), 'format_invalid': set()}

        if 'Transaction ID' not in df.columns:
            return result

        # Define sentinel tokens that should be treated as invalid
        sentinel_tokens = {'DUPLICATE', 'INVALID', 'N/A', 'NULL', 'NA', ''}

        # Canonical Transaction ID pattern: TXN-<digits>
        canonical_pattern = re.compile(r'^TXN-\d+$')

        txn_ids = df['Transaction ID']

        # First pass: identify format violations and sentinels
        for idx, row in df.iterrows():
            txn_id = row['Transaction ID']

            if pd.isna(txn_id):
                continue  # Will be handled by missing check below

            txn_id_str = str(txn_id).strip().upper()

            # Check for sentinel tokens
            if txn_id_str in sentinel_tokens:
                result['format_invalid'].add(idx)
                continue

            # Check for canonical format: TXN-<digits>
            if not canonical_pattern.match(txn_id_str):
                result['format_invalid'].add(idx)
                continue

        # Count non-empty, non-sentinel Transaction IDs
        valid_txn_mask = txn_ids.notna() & ~txn_ids.index.isin(result['format_invalid'])
        valid_count = valid_txn_mask.sum()
        total_count = len(df)

        # If majority (>50%) have valid Transaction ID, enforce it for all
        if valid_count > total_count / 2:
            # Find missing (including sentinels and format violations)
            all_invalid_indices = result['format_invalid'].union(set(df[txn_ids.isna()].index))
            result['missing'] = all_invalid_indices

            # Find duplicates among valid Transaction IDs only
            valid_txn_ids = df[valid_txn_mask]['Transaction ID']
            if len(valid_txn_ids) > 0:
                duplicated = valid_txn_ids.duplicated(keep=False)
                duplicate_indices = df[valid_txn_mask][duplicated].index
                result['duplicates'] = set(duplicate_indices)

        return result

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        error_by_type = {}
        error_by_column = {}

        for error in self.validation_errors:
            # Count by type
            if error.error_type not in error_by_type:
                error_by_type[error.error_type] = 0
            error_by_type[error.error_type] += 1

            # Count by column
            if error.column_name not in error_by_column:
                error_by_column[error.column_name] = 0
            error_by_column[error.column_name] += 1

        return {
            'total_errors': len(self.validation_errors),
            'quarantined_rows': len(self.quarantine_rows),
            'errors_by_type': error_by_type,
            'errors_by_column': error_by_column,
            'error_categories': {
                cat.value: sum(1 for r in self.quarantine_rows if r.error_category == cat)
                for cat in ErrorCategory
            }
        }
