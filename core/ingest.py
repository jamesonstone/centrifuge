"""CSV ingestion and profiling module for Centrifuge."""

import csv
import hashlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

import chardet
import pandas as pd
import structlog
from uuid_extensions import uuid7
from uuid import UUID

from core.models import Schema
from core.schema_inference import SchemaInferencer, ExperimentalFeatures
from core.config import settings

logger = structlog.get_logger()


class CSVIngester:
    """Handles CSV file ingestion with automatic format detection."""

    # Common delimiters to try
    DELIMITERS = [',', '\t', '|', ';', ':']

    # Common encodings to try if chardet fails
    FALLBACK_ENCODINGS = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'ascii']

    # Maximum rows to sample for delimiter detection
    SAMPLE_ROWS = 100

    def __init__(self, schema: Optional[Schema] = None):
        """
        Initialize CSV ingester.

        Args:
            schema: Optional schema for validation and header mapping
        """
        self.schema = schema
        self.delimiter = None
        self.encoding = None
        self.headers = []
        self.normalized_headers = []
        self.file_hash = None
        self.row_count = 0
        self.experimental = ExperimentalFeatures()

    def compute_file_hash(self, content: bytes) -> str:
        """
        Compute SHA256 hash of file content.

        Args:
            content: Raw file bytes

        Returns:
            Hex string of SHA256 hash
        """
        return hashlib.sha256(content).hexdigest()

    def detect_encoding(self, content: bytes) -> str:
        """
        Detect file encoding using chardet with fallback options.

        Args:
            content: Raw file bytes

        Returns:
            Detected encoding name
        """
        # Try chardet first
        detection = chardet.detect(content)
        if detection and detection.get('confidence', 0) > 0.7:
            encoding = detection['encoding']
            logger.info("Detected encoding",
                       encoding=encoding,
                       confidence=detection['confidence'])
            return encoding

        # Try fallback encodings
        for encoding in self.FALLBACK_ENCODINGS:
            try:
                content.decode(encoding)
                logger.info("Using fallback encoding", encoding=encoding)
                return encoding
            except (UnicodeDecodeError, AttributeError):
                continue

        # Default to utf-8 with error handling
        logger.warning("Could not detect encoding, defaulting to utf-8")
        return 'utf-8'

    def detect_delimiter(self, text_sample: str) -> str:
        """
        Detect CSV delimiter by analyzing sample rows.

        Args:
            text_sample: First few lines of the CSV

        Returns:
            Detected delimiter character
        """
        # Try each delimiter and count consistency
        delimiter_scores = {}

        for delimiter in self.DELIMITERS:
            try:
                # Use csv.Sniffer first
                sniffer = csv.Sniffer()
                detected = sniffer.sniff(text_sample, delimiters=delimiter)
                if detected.delimiter == delimiter:
                    delimiter_scores[delimiter] = 1000  # High score for Sniffer match
            except Exception:
                pass

            # Manual scoring based on consistency
            lines = text_sample.strip().split('\n')[:10]
            if len(lines) < 2:
                continue

            counts = [line.count(delimiter) for line in lines]
            if counts and all(c == counts[0] and c > 0 for c in counts):
                # Consistent count across lines
                delimiter_scores[delimiter] = counts[0] * 10
            elif counts and counts[0] > 0:
                # Some inconsistency but delimiter present
                delimiter_scores[delimiter] = counts[0]

        if delimiter_scores:
            best_delimiter = max(delimiter_scores, key=delimiter_scores.get)
            logger.info("Detected delimiter",
                       delimiter=repr(best_delimiter),
                       score=delimiter_scores[best_delimiter])
            return best_delimiter

        # Default to comma
        logger.warning("Could not detect delimiter, defaulting to comma")
        return ','

    def normalize_header(self, header: str) -> str:
        """
        Normalize a header name to canonical form.

        Args:
            header: Raw header string

        Returns:
            Normalized header string
        """
        # Remove quotes if present
        header = header.strip('"\'')

        # Strip whitespace
        header = header.strip()

        # Check schema for aliases
        if self.schema and self.schema.header_aliases:
            # Case-insensitive alias lookup
            for alias, canonical in self.schema.header_aliases.items():
                if alias.lower() == header.lower():
                    return canonical

        # Standard normalization
        # Convert to title case for consistency
        words = header.replace('_', ' ').replace('-', ' ').split()
        normalized = ' '.join(word.capitalize() for word in words)

        # Special cases for common accounting columns
        replacements = {
            'Txn Id': 'Transaction ID',
            'Trans Id': 'Transaction ID',
            'Transaction Id': 'Transaction ID',
            'Acct Code': 'Account Code',
            'Account Cd': 'Account Code',
            'Acct Name': 'Account Name',
            'Account Nm': 'Account Name',
            'Dept': 'Department',
            'Debit Amt': 'Debit Amount',
            'Credit Amt': 'Credit Amount',
            'Ref Number': 'Reference Number',
            'Ref Num': 'Reference Number',
            'Ref No': 'Reference Number',
            'Created Date': 'Created By',
            'Created User': 'Created By',
            'User': 'Created By'
        }

        for pattern, replacement in replacements.items():
            if normalized.lower() == pattern.lower():
                return replacement

        return normalized

    def validate_headers(self, headers: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate headers against schema if provided.

        Args:
            headers: List of normalized headers

        Returns:
            Tuple of (is_valid, missing_required_columns)
        """
        if not self.schema:
            return True, []

        missing = []
        # Create case-insensitive lookup of headers
        headers_lower = {h.lower(): h for h in headers}

        for col_name, col_def in self.schema.columns.items():
            if col_def.required:
                # Try exact match first, then case-insensitive match
                if col_name not in headers and col_name.lower() not in headers_lower:
                    missing.append(col_name)

        return len(missing) == 0, missing

    def ingest_file(
        self,
        file_path: Optional[Path] = None,
        file_content: Optional[bytes] = None,
        s3_url: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Ingest CSV file from various sources.

        Args:
            file_path: Local file path
            file_content: Raw file bytes
            s3_url: S3 URL (requires boto3 setup)

        Returns:
            Pandas DataFrame with normalized headers and row UUIDs
        """
        # Get content based on source
        if file_path:
            with open(file_path, 'rb') as f:
                content = f.read()
            source_name = file_path.name
        elif file_content:
            content = file_content
            source_name = "uploaded_file.csv"
        elif s3_url:
            content = self._download_from_s3(s3_url)
            source_name = s3_url.split('/')[-1]
        else:
            raise ValueError("No input source provided")

        # Compute hash
        self.file_hash = self.compute_file_hash(content)
        logger.info("File hash computed", hash=self.file_hash, source=source_name)

        # Detect encoding
        self.encoding = self.detect_encoding(content)

        # Decode content
        try:
            text_content = content.decode(self.encoding, errors='replace')
        except Exception as e:
            logger.error("Failed to decode file", encoding=self.encoding, error=str(e))
            # Try with error handling
            text_content = content.decode('utf-8', errors='ignore')

        # Clean up any UTF-8 BOM if present
        if text_content.startswith('\ufeff'):
            text_content = text_content[1:]

        # Get sample for delimiter detection
        sample_lines = '\n'.join(text_content.split('\n')[:self.SAMPLE_ROWS])

        # Detect delimiter
        self.delimiter = self.detect_delimiter(sample_lines)

        # Parse CSV with pandas
        try:
            # Use StringIO to read from string
            df = pd.read_csv(
                io.StringIO(text_content),
                sep=self.delimiter,
                encoding=None,  # Already decoded
                dtype=str,  # Read everything as string initially
                na_values=['', 'NA', 'N/A', 'NULL', 'null', 'None'],
                keep_default_na=True,
                skipinitialspace=True,
                quotechar='"',
                thousands=',',  # Handle thousand separators
                engine='python'  # More flexible parsing
            )
        except Exception as e:
            logger.error("Failed to parse CSV", error=str(e))
            # Try with more lenient settings
            df = pd.read_csv(
                io.StringIO(text_content),
                sep=self.delimiter,
                encoding=None,
                dtype=str,
                na_values=[],
                keep_default_na=False,
                on_bad_lines='warn',
                engine='python'
            )

        # Store original headers
        self.headers = list(df.columns)

        # Normalize headers
        self.normalized_headers = [self.normalize_header(h) for h in self.headers]
        df.columns = self.normalized_headers

        # Validate headers
        is_valid, missing = self.validate_headers(self.normalized_headers)
        if not is_valid:
            logger.warning("Missing required columns", missing=missing)

        # Add row UUID for tracking
        # Generate deterministic UUIDs based on row content for reproducibility
        def generate_row_uuid(row_data):
            """Generate deterministic UUID based on row content."""
            # Create a stable hash from row content
            row_str = '|'.join(str(v) for v in row_data.values)
            row_hash = hashlib.sha256(row_str.encode()).hexdigest()
            # Convert first 16 bytes of hash to UUID format
            # This creates a version 5 UUID-like format but deterministic
            uuid_hex = row_hash[:32]
            formatted_uuid = f"{uuid_hex[:8]}-{uuid_hex[8:12]}-{uuid_hex[12:16]}-{uuid_hex[16:20]}-{uuid_hex[20:32]}"
            return formatted_uuid
        
        df['row_uuid'] = df.apply(generate_row_uuid, axis=1)

        # Add original row number (1-indexed for user reference)
        df['row_number'] = range(1, len(df) + 1)

        # Store row count
        self.row_count = len(df)

        logger.info("CSV ingested successfully",
                   rows=self.row_count,
                   columns=len(self.normalized_headers),
                   delimiter=repr(self.delimiter),
                   encoding=self.encoding)

        return df

    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Ingest CSV file (backwards compatibility method).

        Args:
            file_path: Path to CSV file

        Returns:
            Pandas DataFrame with normalized headers and row UUIDs
        """
        return self.ingest_file(file_path=Path(file_path))

    def _download_from_s3(self, s3_url: str) -> bytes:
        """
        Download file from S3.

        Args:
            s3_url: S3 URL (s3://bucket/key)

        Returns:
            File content as bytes
        """
        import boto3
        from urllib.parse import urlparse

        # Parse S3 URL
        parsed = urlparse(s3_url)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')

        # Create S3 client
        s3 = boto3.client(
            's3',
            endpoint_url=settings.artifact_endpoint if 'minio' in settings.artifact_endpoint else None,
            aws_access_key_id=settings.artifact_access_key,
            aws_secret_access_key=settings.artifact_secret_key,
            region_name=settings.artifact_region
        )

        # Download file
        response = s3.get_object(Bucket=bucket, Key=key)
        return response['Body'].read()

    def profile_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Profile the ingested data for statistics.

        Args:
            df: Ingested DataFrame

        Returns:
            Dictionary of profiling statistics
        """
        profile = {
            'row_count': len(df),
            'column_count': len(df.columns) - 2,  # Exclude row_uuid and row_number
            'file_hash': self.file_hash,
            'delimiter': self.delimiter,
            'encoding': self.encoding,
            'columns': {},
            'missing_values': {},
            'unique_counts': {},
            'data_types': {}
        }

        for col in df.columns:
            if col in ['row_uuid', 'row_number']:
                continue

            col_data = df[col]
            profile['columns'][col] = {
                'original_name': self.headers[self.normalized_headers.index(col)]
                    if col in self.normalized_headers else col,
                'normalized_name': col,
                'non_null_count': col_data.notna().sum(),
                'null_count': col_data.isna().sum(),
                'unique_count': col_data.nunique(),
                'most_common': col_data.value_counts().head(5).to_dict() if not col_data.empty else {}
            }

            profile['missing_values'][col] = col_data.isna().sum()
            profile['unique_counts'][col] = col_data.nunique()

            # Infer data type
            if col_data.notna().any():
                sample = col_data.dropna().iloc[0] if not col_data.dropna().empty else None
                if sample:
                    # Try to detect type
                    if re.match(r'^\d{4}-\d{2}-\d{2}', str(sample)):
                        profile['data_types'][col] = 'date'
                    elif re.match(r'^-?\d+\.?\d*$', str(sample).replace(',', '')):
                        profile['data_types'][col] = 'numeric'
                    else:
                        profile['data_types'][col] = 'string'
                else:
                    profile['data_types'][col] = 'unknown'
            else:
                profile['data_types'][col] = 'empty'

        return profile


class HeaderMapper:
    """Maps various header formats to canonical names."""

    # Common variations for accounting columns
    HEADER_MAPPINGS = {
        # Transaction ID variations
        'transaction_id': 'Transaction ID',
        'trans_id': 'Transaction ID',
        'txn_id': 'Transaction ID',
        'transaction id': 'Transaction ID',
        'trans id': 'Transaction ID',
        'txn id': 'Transaction ID',
        'transactionid': 'Transaction ID',
        'trans-id': 'Transaction ID',
        'transaction-id': 'Transaction ID',
        'transaction_number': 'Transaction ID',
        'trans_number': 'Transaction ID',
        'trans_no': 'Transaction ID',
        'transaction_no': 'Transaction ID',

        # Date variations
        'date': 'Date',
        'transaction_date': 'Date',
        'trans_date': 'Date',
        'txn_date': 'Date',
        'posting_date': 'Date',
        'value_date': 'Date',
        'effective_date': 'Date',

        # Account Code variations
        'account_code': 'Account Code',
        'acct_code': 'Account Code',
        'account_cd': 'Account Code',
        'acct_cd': 'Account Code',
        'acc_code': 'Account Code',
        'accountcode': 'Account Code',
        'account-code': 'Account Code',
        'gl_code': 'Account Code',
        'gl_account': 'Account Code',

        # Account Name variations
        'account_name': 'Account Name',
        'acct_name': 'Account Name',
        'account_nm': 'Account Name',
        'acct_nm': 'Account Name',
        'accountname': 'Account Name',
        'account-name': 'Account Name',
        'gl_name': 'Account Name',
        'gl_account_name': 'Account Name',

        # Description variations
        'description': 'Description',
        'desc': 'Description',
        'transaction_description': 'Description',
        'trans_desc': 'Description',
        'narrative': 'Description',
        'particulars': 'Description',
        'details': 'Description',

        # Debit Amount variations
        'debit_amount': 'Debit Amount',
        'debit_amt': 'Debit Amount',
        'debit': 'Debit Amount',
        'dr_amount': 'Debit Amount',
        'dr_amt': 'Debit Amount',
        'dr': 'Debit Amount',
        'debitamount': 'Debit Amount',
        'debit-amount': 'Debit Amount',

        # Credit Amount variations
        'credit_amount': 'Credit Amount',
        'credit_amt': 'Credit Amount',
        'credit': 'Credit Amount',
        'cr_amount': 'Credit Amount',
        'cr_amt': 'Credit Amount',
        'cr': 'Credit Amount',
        'creditamount': 'Credit Amount',
        'credit-amount': 'Credit Amount',

        # Department variations
        'department': 'Department',
        'dept': 'Department',
        'cost_center': 'Department',
        'cost_centre': 'Department',
        'cc': 'Department',
        'division': 'Department',
        'business_unit': 'Department',
        'bu': 'Department',

        # Reference Number variations
        'reference_number': 'Reference Number',
        'ref_number': 'Reference Number',
        'ref_no': 'Reference Number',
        'ref_num': 'Reference Number',
        'reference': 'Reference Number',
        'ref': 'Reference Number',
        'invoice_number': 'Reference Number',
        'invoice_no': 'Reference Number',
        'document_number': 'Reference Number',
        'doc_no': 'Reference Number',

        # Created By variations
        'created_by': 'Created By',
        'created_user': 'Created By',
        'user': 'Created By',
        'username': 'Created By',
        'entered_by': 'Created By',
        'posted_by': 'Created By',
        'modified_by': 'Created By',
        'last_modified_by': 'Created By',
    }

    @classmethod
    def map_header(cls, header: str) -> str:
        """
        Map a header to its canonical name.

        Args:
            header: Raw header string

        Returns:
            Canonical header name
        """
        # Clean and normalize for lookup
        clean = header.lower().strip().replace(' ', '_').replace('-', '_')

        # Remove common prefixes/suffixes
        clean = re.sub(r'^(tbl_|col_|fld_)', '', clean)
        clean = re.sub(r'(_col|_fld|_field)$', '', clean)

        # Look up in mappings
        if clean in cls.HEADER_MAPPINGS:
            return cls.HEADER_MAPPINGS[clean]

        # Return original if no mapping found
        return header


class DataIngestor:
    """High-level data ingestor with experimental features support."""

    def __init__(self, schema: Optional[Schema] = None):
        """
        Initialize data ingestor.

        Args:
            schema: Optional schema (can be inferred experimentally)
        """
        self.schema = schema
        self.csv_ingester = CSVIngester(schema)
        self.experimental = ExperimentalFeatures()

    def ingest_csv(
        self,
        file_path: str,
        use_inferred: bool = False,
        acknowledge_experimental: bool = False
    ) -> Tuple[pd.DataFrame, str, Dict[str, str]]:
        """
        Ingest CSV file with optional schema inference.

        Args:
            file_path: Path to CSV file
            use_inferred: Enable experimental schema inference
            acknowledge_experimental: Acknowledge experimental features

        Returns:
            Tuple of (dataframe, file_hash, header_mapping)
        """
        if use_inferred:
            if not acknowledge_experimental:
                raise ValueError(
                    "Must acknowledge experimental features with acknowledge_experimental=True"
                )

            logger.warning("EXPERIMENTAL: Schema inference enabled")
            self.experimental.enable_schema_inference()

            # Ingest without schema first
            temp_ingester = CSVIngester()
            df = temp_ingester.ingest(file_path)

            # Infer schema
            inferencer = SchemaInferencer()
            self.schema = inferencer.infer_schema(df, use_experimental=True)

            # Re-ingest with inferred schema
            self.csv_ingester = CSVIngester(self.schema)
            df = self.csv_ingester.ingest(file_path)

            logger.info("Experimental ingest complete",
                       schema_version=self.schema.version,
                       columns=len(self.schema.columns))
        else:
            # Standard ingestion
            df = self.csv_ingester.ingest(file_path)

        # Build header mapping
        header_mapping = {}
        for orig, norm in zip(self.csv_ingester.headers, self.csv_ingester.normalized_headers):
            if orig != norm:
                header_mapping[orig] = norm

        return df, self.csv_ingester.file_hash, header_mapping

    def set_llm_columns(
        self,
        columns: List[str],
        force: bool = False
    ) -> None:
        """
        Set which columns can use LLM (experimental).

        Args:
            columns: List of column names
            force: Force even if not in allowed list
        """
        if not force:
            allowed = ['department', 'account_name', 'category', 'type', 'status']
            for col in columns:
                if col.lower() not in allowed:
                    raise ValueError(
                        f"Column '{col}' not in allowed LLM columns. Use force=True to override."
                    )

        self.experimental.set_llm_columns(columns)
        logger.warning(f"EXPERIMENTAL: LLM enabled for columns: {columns}")

    def set_column_prompt_permission(
        self,
        column: str,
        allow: bool
    ) -> None:
        """
        Set whether column can be included in prompts (experimental).

        Args:
            column: Column name
            allow: Whether to allow in prompts
        """
        self.experimental.set_column_prompt_permission(column, allow)

        # Update schema if available
        if self.schema:
            for col_def in self.schema.columns:
                if col_def.name == column or col_def.display_name == column:
                    col_def.allow_in_prompt = allow
                    break

    def get_experimental_config(self) -> Dict[str, Any]:
        """Get current experimental configuration."""
        return self.experimental.get_configuration()
