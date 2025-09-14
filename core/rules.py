"""Rules engine for deterministic data transformations."""

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import pandas as pd
import structlog
from dateutil import parser as date_parser

from core.models import (
    Schema, ColumnType, Patch, SourceType,
    AuditEvent, DiffEntry
)
from core.validation import Validator

logger = structlog.get_logger()


class RuleEngine:
    """Apply deterministic transformation rules to data."""

    # Date formats in order of preference for output
    CANONICAL_DATE_FORMAT = '%Y-%m-%d'  # ISO format

    # Date parsing formats (same as validator)
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
        '%d/%m/%Y',           # 10/02/2024
        '%Y-%m-%d',           # 2024-02-11
        '%d %b %Y',           # 21st Feb 2024 (with ordinal)
        '%d %B %Y',           # 21st February 2024
    ]

    def __init__(self, schema: Schema):
        """
        Initialize rules engine with schema.

        Args:
            schema: Schema defining data structure
        """
        self.schema = schema
        self.patches: List[Patch] = []
        self.audit_events: List[AuditEvent] = []
        self.diff_entries: List[DiffEntry] = []

    def apply_rules(
        self,
        df: pd.DataFrame,
        run_id: UUID
    ) -> Tuple[pd.DataFrame, List[Patch], List[AuditEvent], List[DiffEntry]]:
        """
        Apply all deterministic rules to the DataFrame.

        Args:
            df: DataFrame to transform
            run_id: Run ID for audit tracking

        Returns:
            Tuple of (transformed_df, patches, audit_events, diff_entries)
        """
        logger.info("Starting rules engine", rows=len(df))

        # Reset tracking
        self.patches = []
        self.audit_events = []
        self.diff_entries = []

        # Create working copy
        working_df = df.copy()

        # Apply rules in order

        # 1. Trim whitespace from all string columns
        working_df = self._apply_trim_whitespace(working_df, run_id)

        # 2. Normalize case for categorical columns
        working_df = self._apply_case_normalization(working_df, run_id)

        # 3. Clean and typecast numeric fields
        working_df = self._apply_numeric_cleaning(working_df, run_id)

        # 4. Normalize dates to ISO format
        working_df = self._apply_date_normalization(working_df, run_id)

        # 5. Apply enum mappings for known variations
        working_df = self._apply_enum_mappings(working_df, run_id)

        # 6. Fix Transaction ID formatting
        working_df = self._apply_transaction_id_rules(working_df, run_id)

        # 7. Fix debit/credit sign issues
        working_df = self._apply_debit_credit_fixes(working_df, run_id)

        # 8. Clean special characters from text fields
        working_df = self._apply_text_cleaning(working_df, run_id)

        # 9. Normalize Reference Numbers
        working_df = self._apply_reference_number_rules(working_df, run_id)

        logger.info("Rules engine complete",
                   patches=len(self.patches),
                   audit_events=len(self.audit_events))

        return working_df, self.patches, self.audit_events, self.diff_entries

    def _apply_trim_whitespace(self, df: pd.DataFrame, run_id: UUID) -> pd.DataFrame:
        """Trim leading/trailing whitespace from string columns."""
        for col in df.columns:
            if col in ['row_uuid', 'row_number']:
                continue

            # Only process string columns
            if self.schema.columns.get(col) and \
               self.schema.columns[col].type == ColumnType.STRING:
                for idx, row in df.iterrows():
                    value = row[col]
                    if pd.notna(value) and isinstance(value, str):
                        trimmed = value.strip()
                        # Also collapse internal whitespace
                        trimmed = ' '.join(trimmed.split())

                        if trimmed != value:
                            self._record_change(
                                df, idx, col, value, trimmed, run_id,
                                rule_id='trim_whitespace',
                                reason='Removed extra whitespace'
                            )
                            df.at[idx, col] = trimmed
        return df

    def _apply_case_normalization(self, df: pd.DataFrame, run_id: UUID) -> pd.DataFrame:
        """Normalize case for categorical text columns."""
        # Columns that should be title case
        title_case_columns = ['Department', 'Account Name']

        for col in title_case_columns:
            if col not in df.columns:
                continue

            for idx, row in df.iterrows():
                value = row[col]
                if pd.notna(value) and isinstance(value, str):
                    # Convert to title case
                    normalized = value.strip().title()

                    # Special handling for common abbreviations
                    replacements = {
                        'It': 'IT',
                        'Hr': 'HR',
                        'Ops': 'Operations',
                        'Admin': 'Admin',  # Keep as is
                        'Mkt': 'Marketing'
                    }

                    for old, new in replacements.items():
                        if normalized == old:
                            normalized = new
                            break

                    if normalized != value:
                        self._record_change(
                            df, idx, col, value, normalized, run_id,
                            rule_id='case_normalization',
                            reason='Normalized to title case'
                        )
                        df.at[idx, col] = normalized

        # Uppercase for codes
        code_columns = ['Account Code', 'Transaction ID']
        for col in code_columns:
            if col not in df.columns:
                continue

            for idx, row in df.iterrows():
                value = row[col]
                if pd.notna(value) and isinstance(value, str):
                    upper = value.strip().upper()
                    if upper != value:
                        self._record_change(
                            df, idx, col, value, upper, run_id,
                            rule_id='uppercase_codes',
                            reason='Converted to uppercase'
                        )
                        df.at[idx, col] = upper

        return df

    def _apply_numeric_cleaning(self, df: pd.DataFrame, run_id: UUID) -> pd.DataFrame:
        """Clean and normalize numeric fields."""
        # Find numeric columns (case-insensitive)
        numeric_columns = []
        for col in df.columns:
            col_lower = col.lower().replace('_', ' ').replace('-', ' ')
            if any(keyword in col_lower for keyword in ['debit', 'credit', 'amount']):
                numeric_columns.append(col)

        # Convert numeric columns to object type to allow string storage
        for col in numeric_columns:
            df[col] = df[col].astype('object')

        for col in numeric_columns:
            for idx, row in df.iterrows():
                value = row[col]
                # Handle both NaN and non-NaN values
                cleaned = self._clean_numeric_value(value)

                # For monetary fields, keep as formatted decimal string to preserve precision
                # Convert to float only for comparison
                try:
                    cleaned_float = float(cleaned)
                except (ValueError, TypeError):
                    cleaned_float = 0.0
                    cleaned = '0.00'

                # Check if value actually changed (avoid formatting already formatted values)
                # For floats, check if they're effectively the same value
                original_float = None
                try:
                    if pd.notna(value) and value != '':
                        original_float = float(value) if not isinstance(value, str) else float(self._clean_numeric_value(value))
                except (ValueError, TypeError):
                    original_float = None

                # Only record change if the values are truly different
                if original_float is not None and abs(original_float - cleaned_float) < 0.001:
                    # Values are effectively the same, don't record a change
                    # But store as formatted decimal string for currency precision
                    df.at[idx, col] = cleaned
                elif str(cleaned) != str(value):
                    self._record_change(
                        df, idx, col, value, cleaned, run_id,
                        rule_id='numeric_cleaning',
                        reason='Cleaned numeric formatting'
                    )
                    # Store as formatted decimal string for currency precision
                    df.at[idx, col] = cleaned
                else:
                    # No change needed, but store as formatted decimal string
                    df.at[idx, col] = cleaned

        return df

    def _clean_numeric_value(self, value: Any) -> str:
        """
        Clean a numeric value to standard format.

        Args:
            value: Raw numeric value

        Returns:
            Cleaned numeric string
        """
        if pd.isna(value) or value == '' or value is None:
            return '0.00'

        try:
            # Convert to string
            value_str = str(value).strip()

            # Remove quotes
            value_str = value_str.replace('"', '').replace("'", '')

            # Remove currency symbols
            value_str = value_str.replace('$', '').replace('£', '').replace('€', '')

            # Remove commas
            value_str = value_str.replace(',', '')

            # Handle parentheses for negative values
            if value_str.startswith('(') and value_str.endswith(')'):
                value_str = '-' + value_str[1:-1]

            # Parse to Decimal for validation
            if value_str and value_str != '-':
                decimal_value = Decimal(value_str)
                # Format with 2 decimal places
                return f"{decimal_value:.2f}"
            else:
                return '0.00'

        except (InvalidOperation, ValueError):
            # Return 0.00 if can't parse
            return '0.00'

    def _apply_date_normalization(self, df: pd.DataFrame, run_id: UUID) -> pd.DataFrame:
        """Normalize dates to ISO format (YYYY-MM-DD)."""
        # Find date columns (case-insensitive)
        date_columns = []
        for col in df.columns:
            if col.lower() in ['date']:
                date_columns.append(col)

        for col in date_columns:

            for idx, row in df.iterrows():
                value = row[col]
                if pd.notna(value) and value:
                    normalized = self._normalize_date(value)
                    if normalized and normalized != value:
                        self._record_change(
                            df, idx, col, value, normalized, run_id,
                            rule_id='date_normalization',
                            reason='Normalized to ISO date format'
                        )
                        df.at[idx, col] = normalized

        return df

    def _normalize_date(self, value: Any) -> Optional[str]:
        """
        Normalize a date value to ISO format.

        Args:
            value: Raw date value

        Returns:
            ISO formatted date string or None
        """
        if not value or str(value).lower() in ['null', 'na', 'n/a']:
            return None

        value_str = str(value).strip()

        # Remove ordinal indicators (1st, 2nd, 3rd, etc.)
        value_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', value_str)

        # Try specific formats first
        for date_format in self.DATE_FORMATS:
            try:
                parsed_date = datetime.strptime(value_str, date_format)
                return parsed_date.strftime(self.CANONICAL_DATE_FORMAT)
            except ValueError:
                continue

        # Try dateutil parser as fallback
        try:
            parsed_date = date_parser.parse(value_str, fuzzy=False)
            return parsed_date.strftime(self.CANONICAL_DATE_FORMAT)
        except (ValueError, TypeError):
            pass

        # Special handling for common variations
        # Handle "Jan 26, 2024" format (with comma)
        if ',' in value_str:
            try:
                parsed_date = date_parser.parse(value_str)
                return parsed_date.strftime(self.CANONICAL_DATE_FORMAT)
            except:
                pass

        return None

    def _apply_enum_mappings(self, df: pd.DataFrame, run_id: UUID) -> pd.DataFrame:
        """Apply enum mappings for known variations."""

        # Department mappings (rules-based, not LLM)
        department_mappings = {
            # Sales variations
            'SALES': 'Sales',
            'sales': 'Sales',
            'Sales': 'Sales',

            # Operations variations
            'Operations': 'Operations',
            'operations': 'Operations',
            'OPERATIONS': 'Operations',
            'OPS': 'Operations',
            'Ops': 'Operations',
            'ops': 'Operations',

            # Admin variations
            'Admin': 'Admin',
            'ADMIN': 'Admin',
            'admin': 'Admin',
            'Administration': 'Admin',

            # IT variations
            'IT': 'IT',
            'It': 'IT',
            'Information Technology': 'IT',
            'Tech': 'IT',
            'Technology': 'IT',

            # Finance variations
            'Finance': 'Finance',
            'FINANCE': 'Finance',
            'finance': 'Finance',
            'Treasury': 'Finance',

            # Marketing variations
            'Marketing': 'Marketing',
            'marketing': 'Marketing',
            'MARKETING': 'Marketing',
            'MKT': 'Marketing',
            'Mkt': 'Marketing',

            # HR variations
            'HR': 'HR',
            'Hr': 'HR',
            'hr': 'HR',
            'Human Resources': 'HR',

            # Other common departments
            'Legal': 'Legal',
            'Engineering': 'Engineering',
            'Support': 'Support',
            'Maintenance': 'Operations',
            'Facilities': 'Operations',
            'General': 'Admin'
        }

        # Apply department mappings
        dept_col = None
        for col in df.columns:
            if col.lower() in ['department', 'dept']:
                dept_col = col
                break

        if dept_col:
            for idx, row in df.iterrows():
                value = row[dept_col]
                if pd.notna(value) and value in department_mappings:
                    mapped = department_mappings[value]
                    if mapped != value:
                        self._record_change(
                            df, idx, dept_col, value, mapped, run_id,
                            rule_id='department_mapping',
                            reason=f'Mapped department variant to canonical form'
                        )
                        df.at[idx, dept_col] = mapped

        # Account Name mappings (basic rules, complex ones go to LLM)
        account_mappings = {
            'cash': 'Cash',
            'CASH': 'Cash',
            'Bank': 'Cash',
            'Accounts Receivable': 'Accounts Receivable',
            'accounts receivable': 'Accounts Receivable',
            'A/R': 'Accounts Receivable',
            'Accounts Payable': 'Accounts Payable',
            'accounts payable': 'Accounts Payable',
            'A/P': 'Accounts Payable',
            'Cost of Goods Sold': 'Cost of Goods Sold',
            'COGS': 'Cost of Goods Sold',
            'Sales Revenue': 'Sales Revenue',
            'Revenue': 'Sales Revenue',
            'Equipment': 'Equipment',
            'Office Supplies': 'Operating Expenses',
            'Utilities Expense': 'Operating Expenses',
            'Rent Expense': 'Operating Expenses',
            'Professional Fees': 'Operating Expenses',
            'Travel & Entertainment': 'Operating Expenses'
        }

        account_col = None
        for col in df.columns:
            if col.lower().replace(' ', '_').replace('-', '_') in ['account_name', 'account']:
                account_col = col
                break

        if account_col:
            for idx, row in df.iterrows():
                value = row[account_col]
                if pd.notna(value):
                    # Try exact match first
                    if value in account_mappings:
                        mapped = account_mappings[value]
                        if mapped != value:
                            self._record_change(
                                df, idx, account_col, value, mapped, run_id,
                                rule_id='account_mapping',
                                reason='Mapped account variant to canonical form'
                            )
                            df.at[idx, account_col] = mapped
                    else:
                        # If no exact match, apply title case normalization
                        # but keep common abbreviations uppercase
                        original_value = str(value).strip()
                        title_cased = original_value.title()

                        # Handle special cases for accounting terms
                        title_cased = title_cased.replace(' Of ', ' of ')
                        title_cased = title_cased.replace(' And ', ' and ')
                        title_cased = title_cased.replace(' The ', ' the ')
                        title_cased = title_cased.replace(' A ', ' a ')
                        title_cased = title_cased.replace(' An ', ' an ')

                        if title_cased != original_value:
                            self._record_change(
                                df, idx, account_col, original_value, title_cased, run_id,
                                rule_id='account_title_case',
                                reason='Applied title case to account name'
                            )
                            df.at[idx, account_col] = title_cased

        return df

    def _apply_transaction_id_rules(self, df: pd.DataFrame, run_id: UUID) -> pd.DataFrame:
        """Fix Transaction ID formatting issues with canonical hyphen format and zero-padding."""
        if 'Transaction ID' not in df.columns:
            return df

        for idx, row in df.iterrows():
            value = row['Transaction ID']
            if pd.notna(value) and value:
                original = str(value)
                cleaned = original.strip().upper()

                # Remove internal spaces
                cleaned = cleaned.replace(' ', '')

                # Handle special cases first - leave sentinel tokens unchanged for validation
                sentinel_tokens = {'DUPLICATE', 'INVALID', 'N/A', 'NULL', 'NA'}
                if cleaned.upper() in sentinel_tokens:
                    # This will be caught in validation
                    continue

                # Normalize prefix handling
                if cleaned.startswith('TXN-') or cleaned.startswith('TXN_'):
                    # Extract numeric part after prefix
                    numeric_part = cleaned[4:]
                elif cleaned.startswith('TXN'):
                    # Extract numeric part after TXN
                    numeric_part = cleaned[3:]
                elif cleaned and cleaned[0].isdigit():
                    # Add TXN prefix if missing for digit-starting values
                    numeric_part = cleaned
                else:
                    # For non-digit starting values that don't have TXN prefix,
                    # leave unchanged for validation to catch
                    continue

                # Extract contiguous digits from numeric part
                digit_match = re.search(r'\d+', numeric_part)
                if digit_match:
                    digits = digit_match.group()

                    # Apply zero-padding to minimum 3 digits, preserving existing leading zeros
                    if len(digits) < 3:
                        padded_digits = digits.zfill(3)
                    else:
                        padded_digits = digits  # Preserve existing leading zeros for ≥3 digits

                    # Format with canonical hyphen
                    formatted = f'TXN-{padded_digits}'

                    if formatted != original:
                        self._record_change(
                            df, idx, 'Transaction ID', original, formatted, run_id,
                            rule_id='transaction_id_format',
                            reason='Standardized Transaction ID format'
                        )
                        df.at[idx, 'Transaction ID'] = formatted
                else:
                    # No contiguous digits found - leave unchanged for validation to quarantine
                    continue

        return df

    def _apply_debit_credit_fixes(self, df: pd.DataFrame, run_id: UUID) -> pd.DataFrame:
        """Fix debit/credit sign issues and enforce XOR rule."""

        for idx, row in df.iterrows():
            debit = row.get('Debit Amount')
            credit = row.get('Credit Amount')

            # Parse amounts
            debit_val = self._parse_amount_to_decimal(debit)
            credit_val = self._parse_amount_to_decimal(credit)

            # Fix negative debit (should be credit)
            if debit_val is not None and debit_val < 0:
                new_credit = str(abs(debit_val))
                self._record_change(
                    df, idx, 'Debit Amount', debit, '', run_id,
                    rule_id='debit_sign_fix',
                    reason='Moved negative debit to credit'
                )
                self._record_change(
                    df, idx, 'Credit Amount', credit or '', new_credit, run_id,
                    rule_id='debit_sign_fix',
                    reason='Moved negative debit to credit'
                )
                df.at[idx, 'Debit Amount'] = ''
                df.at[idx, 'Credit Amount'] = new_credit

            # Fix negative credit (should be debit)
            elif credit_val is not None and credit_val < 0:
                new_debit = str(abs(credit_val))
                self._record_change(
                    df, idx, 'Credit Amount', credit, '', run_id,
                    rule_id='credit_sign_fix',
                    reason='Moved negative credit to debit'
                )
                self._record_change(
                    df, idx, 'Debit Amount', debit or '', new_debit, run_id,
                    rule_id='credit_sign_fix',
                    reason='Moved negative credit to debit'
                )
                df.at[idx, 'Credit Amount'] = ''
                df.at[idx, 'Debit Amount'] = new_debit

            # Fix both zero - can't determine which should be set
            elif (debit_val == 0 or debit == '0') and (credit_val == 0 or credit == '0'):
                # This will be caught in validation
                pass

        return df

    def _parse_amount_to_decimal(self, value: Any) -> Optional[Decimal]:
        """Parse amount to Decimal for comparison."""
        if pd.isna(value) or value == '' or value is None:
            return None

        try:
            cleaned = self._clean_numeric_value(value)
            if cleaned and cleaned != '':
                return Decimal(cleaned)
        except:
            pass

        return None

    def _apply_text_cleaning(self, df: pd.DataFrame, run_id: UUID) -> pd.DataFrame:
        """Clean special characters from text fields."""
        text_columns = ['Description', 'Created By']

        for col in text_columns:
            if col not in df.columns:
                continue

            for idx, row in df.iterrows():
                value = row[col]
                if pd.notna(value) and value:
                    original = str(value)
                    # Remove problematic special characters but keep basic punctuation
                    cleaned = re.sub(r'[μ∞Ø!]+', '', original)
                    # Remove multiple exclamation marks
                    cleaned = re.sub(r'!+', '!', cleaned)
                    # Clean up spacing
                    cleaned = ' '.join(cleaned.split())

                    if cleaned != original:
                        self._record_change(
                            df, idx, col, original, cleaned, run_id,
                            rule_id='text_cleaning',
                            reason='Removed special characters'
                        )
                        df.at[idx, col] = cleaned

        return df

    def _apply_reference_number_rules(self, df: pd.DataFrame, run_id: UUID) -> pd.DataFrame:
        """Normalize Reference Number formatting."""
        # Find reference number columns (case-insensitive)
        ref_col = None
        for col in df.columns:
            if 'reference' in col.lower().replace('_', ' ').replace('-', ' '):
                ref_col = col
                break

        if not ref_col:
            return df

        for idx, row in df.iterrows():
            value = row[ref_col]
            if pd.notna(value) and value:
                original = str(value)

                # Only normalize if it looks like a reference number (contains REF or looks like it should)
                if 'ref' in original.lower() or re.match(r'^[A-Z0-9]+[-_]?\d+$', original, re.IGNORECASE):
                    # Standardize format
                    cleaned = original.strip().upper()
                    # Replace underscores with hyphens
                    cleaned = cleaned.replace('_', '-')
                    # Remove spaces
                    cleaned = cleaned.replace(' ', '-')
                    # Remove duplicate hyphens
                    cleaned = re.sub(r'-+', '-', cleaned)

                    # Add hyphen after REF if missing
                    if cleaned.startswith('REF') and len(cleaned) > 3 and cleaned[3] != '-':
                        cleaned = 'REF-' + cleaned[3:]

                    if cleaned != original:
                        self._record_change(
                            df, idx, ref_col, original, cleaned, run_id,
                            rule_id='reference_number_format',
                            reason='Standardized reference number format'
                        )
                        df.at[idx, ref_col] = cleaned
                # else: leave non-reference-like values unchanged

        return df

    def _record_change(
        self,
        df: pd.DataFrame,
        idx: int,
        column: str,
        before_value: Any,
        after_value: Any,
        run_id: UUID,
        rule_id: str,
        reason: str
    ):
        """Record a change made by a rule."""
        row = df.loc[idx]
        row_uuid = UUID(row['row_uuid'])
        row_number = row['row_number']

        # Create patch
        patch = Patch(
            row_uuid=row_uuid,
            row_number=row_number,
            column_name=column,
            before_value=str(before_value) if pd.notna(before_value) else None,
            after_value=str(after_value),
            confidence=1.0,  # Rules have 100% confidence
            source=SourceType.RULE,
            rule_id=rule_id,
            reason=reason
        )
        self.patches.append(patch)

        # Create audit event
        audit = AuditEvent(
            run_id=run_id,
            row_uuid=row_uuid,
            column_name=column,
            before_value=str(before_value) if pd.notna(before_value) else None,
            after_value=str(after_value),
            source=SourceType.RULE,
            rule_id=rule_id,
            reason=reason,
            confidence=1.0
        )
        self.audit_events.append(audit)

        # Create diff entry
        diff = DiffEntry(
            row_number=row_number,
            row_uuid=row_uuid,
            column_name=column,
            before_value=str(before_value) if pd.notna(before_value) else None,
            after_value=str(after_value),
            source=SourceType.RULE,
            reason=reason,
            confidence=1.0,  # Rules have 100% confidence
            timestamp=datetime.now()
        )
        self.diff_entries.append(diff)

    def get_rules_summary(self) -> Dict[str, Any]:
        """Get summary of rules applied."""
        rules_count = {}
        columns_affected = {}

        for patch in self.patches:
            # Count by rule
            if patch.rule_id not in rules_count:
                rules_count[patch.rule_id] = 0
            rules_count[patch.rule_id] += 1

            # Count by column
            if patch.column_name not in columns_affected:
                columns_affected[patch.column_name] = 0
            columns_affected[patch.column_name] += 1

        return {
            'total_changes': len(self.patches),
            'rules_applied': rules_count,
            'columns_affected': columns_affected,
            'audit_events': len(self.audit_events),
            'diff_entries': len(self.diff_entries)
        }
