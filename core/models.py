"""Core data models and contracts for Centrifuge."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Literal
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, field_validator
from uuid_extensions import uuid7


class RunStatus(str, Enum):
    """Run execution status."""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


class SourceType(str, Enum):
    """Source of a data change."""
    RULE = "rule"
    LLM = "llm"
    HUMAN = "human"
    CACHE = "cache"


class ErrorCategory(str, Enum):
    """Categories of errors that can occur during processing."""
    VALIDATION_FAILURE = "validation_failure"
    LLM_CONTRACT_FAILURE = "llm_contract_failure"
    LOW_CONFIDENCE = "low_confidence"
    EDIT_CAP_EXCEEDED = "edit_cap_exceeded"
    PARSE_ERROR = "parse_error"


class ColumnType(str, Enum):
    """Data types for columns."""
    STRING = "string"
    DATE = "date"
    DECIMAL = "decimal"
    INTEGER = "integer"
    BOOLEAN = "boolean"


class ColumnPolicy(str, Enum):
    """Policy for how columns are processed."""
    RULE_ONLY = "rule_only"
    LLM_ALLOWED = "llm_allowed"


# Backwards compatibility alias
DataType = ColumnType


# =====================================================================
# SCHEMA CONTRACT
# =====================================================================

class ColumnDefinition(BaseModel):
    """Definition of a single column in the schema."""

    name: str
    display_name: str
    data_type: Optional[ColumnType] = None  # Alias for backwards compatibility
    type: ColumnType
    policy: ColumnPolicy
    required: bool = True
    allow_in_prompt: bool = False
    llm_enabled: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None  # regex pattern
    enum_values: Optional[List[str]] = None
    allowed_values: Optional[List[str]] = None  # Alias for enum_values
    min_value: Optional[Decimal] = None
    max_value: Optional[Decimal] = None
    date_format: Optional[str] = None  # for parsing dates
    is_primary_key: bool = False

    def __init__(self, **data):
        # Handle data_type alias
        if 'data_type' in data and 'type' not in data:
            data['type'] = data['data_type']
        elif 'type' in data and 'data_type' not in data:
            data['data_type'] = data['type']
        # Handle allowed_values alias
        if 'allowed_values' in data and 'enum_values' not in data:
            data['enum_values'] = data['allowed_values']
        super().__init__(**data)


class SchemaConstraints(BaseModel):
    """Constraints that apply across columns."""

    primary_key: Optional[str] = None
    unique: List[str] = Field(default_factory=list)
    debit_xor_credit: bool = False  # Business rule for accounting data
    required_columns: List[str] = Field(default_factory=list)


class DomainRules(BaseModel):
    """Business domain rules."""

    debit_xor_credit: Optional[str] = None
    custom_rules: Dict[str, str] = Field(default_factory=dict)


class Schema(BaseModel):
    """Complete schema definition for CSV validation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID = Field(default_factory=uuid7)
    name: str = ""
    version: str
    columns: Dict[str, ColumnDefinition]  # Dictionary for column name -> definition
    constraints: SchemaConstraints = Field(default_factory=SchemaConstraints)
    header_aliases: Dict[str, str] = Field(default_factory=dict)  # alias -> canonical
    domain_rules: DomainRules = Field(default_factory=DomainRules)
    canonical_departments: List[str] = Field(
        default_factory=lambda: [
            "Sales", "Operations", "Admin", "IT", "Finance",
            "Marketing", "HR", "Legal", "Engineering", "Support"
        ]
    )
    canonical_accounts: List[str] = Field(
        default_factory=lambda: [
            "Cash", "Accounts Receivable", "Accounts Payable",
            "Sales Revenue", "Cost of Goods Sold", "Operating Expenses",
            "Equipment", "Inventory", "Retained Earnings"
        ]
    )
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def __init__(self, **data):
        # Handle case where columns is passed as a list (for test compatibility)
        if 'columns' in data and isinstance(data['columns'], list):
            # Convert list of ColumnDefinition to dict
            columns_dict = {}
            for col_def in data['columns']:
                if isinstance(col_def, ColumnDefinition):
                    # Use name as key
                    columns_dict[col_def.name] = col_def
                elif isinstance(col_def, dict):
                    # Create ColumnDefinition from dict
                    col_obj = ColumnDefinition(**col_def)
                    columns_dict[col_obj.name] = col_obj
            data['columns'] = columns_dict
        elif 'columns' not in data:
            # Default to empty dict
            data['columns'] = {}
        super().__init__(**data)


# =====================================================================
# MANIFEST CONTRACT
# =====================================================================

class RunOptions(BaseModel):
    """Options for a cleaning run."""

    use_inferred: bool = False
    dry_run: bool = False
    llm_columns: List[str] = Field(default_factory=lambda: ["Department", "Account Name"])
    edit_cap_pct: int = 20
    confidence_floor: float = 0.80
    row_limit: int = 50000
    batch_size: int = 15


class Manifest(BaseModel):
    """Run manifest capturing all versions and configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: UUID
    run_seq: int  # Human-friendly sequential number

    # Versioning
    schema_version: str
    model_id: str = "openai/gpt-5"
    prompt_version: str = "1.0.0"
    engine_version: str = "0.1.0"

    # Input tracking
    input_file_name: str
    input_file_hash: str  # SHA256
    input_row_count: int

    # Configuration
    options: RunOptions
    seed: int = 42
    temperature: float = 0.0

    # Timing
    started_at: datetime
    completed_at: Optional[datetime] = None

    # Idempotency
    idempotency_key: str  # hash(input + versions + options)

    def generate_idempotency_key(self) -> str:
        """Generate idempotency key from manifest data."""
        import hashlib
        key_parts = [
            self.input_file_hash,
            self.schema_version,
            self.model_id,
            self.prompt_version,
            str(self.seed),
            str(self.options.model_dump_json())
        ]
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()


# =====================================================================
# PATCH CONTRACT (LLM Response)
# =====================================================================


# Backwards compatibility aliases
RunManifest = Manifest


# =====================================================================
# PATCH CONTRACT (LLM Response)
# =====================================================================

class Patch(BaseModel):
    """A single cell patch from LLM or rule engine."""

    row_uuid: UUID
    row_number: int  # Original row number for reference
    column_name: str
    before_value: Optional[str]
    after_value: str
    confidence: float = 1.0  # 1.0 for rules, variable for LLM
    source: SourceType
    rule_id: Optional[str] = None
    contract_id: Optional[str] = None
    reason: str

    @field_validator('confidence')
    def validate_confidence(cls, v):
        """Ensure confidence is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v


class PatchBatch(BaseModel):
    """Batch of patches from LLM."""

    patches: List[Patch]
    model_id: str
    prompt_version: str
    processing_time_ms: int
    token_count: Optional[int] = None

    def filter_by_confidence(self, min_confidence: float) -> List[Patch]:
        """Filter patches by minimum confidence threshold."""
        return [p for p in self.patches if p.confidence >= min_confidence]


# =====================================================================
# AUDIT CONTRACT
# =====================================================================

class AuditEvent(BaseModel):
    """Audit log entry for a single change."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID = Field(default_factory=uuid7)
    run_id: UUID
    row_uuid: UUID
    column_name: str
    before_value: Optional[str]
    after_value: str
    source: SourceType
    rule_id: Optional[str] = None
    contract_id: Optional[str] = None
    reason: str
    confidence: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.now)

    def to_ndjson_dict(self) -> dict:
        """Convert to dict for NDJSON serialization."""
        return {
            "id": str(self.id),
            "run_id": str(self.run_id),
            "row_uuid": str(self.row_uuid),
            "column_name": self.column_name,
            "before_value": self.before_value,
            "after_value": self.after_value,
            "source": self.source.value,
            "rule_id": self.rule_id,
            "contract_id": self.contract_id,
            "reason": self.reason,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat()
        }


# =====================================================================
# METRICS CONTRACT
# =====================================================================

class Metrics(BaseModel):
    """Run metrics and statistics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: UUID

    # Counts
    total_rows: int
    total_cells: int
    cells_validated: int
    cells_modified: int
    cells_quarantined: int

    # By source
    rules_fixed_count: int = 0
    llm_fixed_count: int = 0
    cache_fixed_count: int = 0

    # Performance
    rules_duration_ms: int
    llm_duration_ms: int
    validation_duration_ms: int
    total_duration_ms: int

    # LLM stats
    llm_calls_count: Optional[int] = None
    llm_tokens_used: Optional[int] = None
    llm_cost_estimate: Optional[Decimal] = None
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_ratio: Optional[float] = None

    # Error breakdown
    validation_failures: int = 0
    llm_contract_failures: int = 0
    low_confidence_count: int = 0
    edit_cap_exceeded_count: int = 0
    parse_errors: int = 0

    # Per-column stats
    column_stats: Dict[str, Dict[str, int]] = Field(default_factory=dict)

    def calculate_cache_hit_ratio(self) -> None:
        """Calculate cache hit ratio."""
        total = self.cache_hits + self.cache_misses
        if total > 0:
            self.cache_hit_ratio = self.cache_hits / total


# Backwards compatibility aliases
RunManifest = Manifest
RunMetrics = Metrics
RunMetrics = Metrics


# =====================================================================================================================================
# SUMMARY CONTRACT
# =====================================================================

class Summary(BaseModel):
    """Human-readable summary of run results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: UUID
    run_seq: int
    status: RunStatus

    # Overview
    title: str = "CSV Cleaning Run Summary"
    description: str
    input_file: str
    started_at: datetime
    completed_at: datetime
    duration_seconds: float

    # Results
    total_rows: int
    cleaned_rows: int
    quarantined_rows: int
    success_rate: float

    # Changes by source
    rules_fixes: Dict[str, int]  # rule_id -> count
    llm_fixes: Dict[str, int]  # column -> count
    cache_fixes: Dict[str, int]  # column -> count

    # Quarantine breakdown
    quarantine_reasons: Dict[ErrorCategory, int]

    # Top issues found
    top_issues: List[Dict[str, Any]]

    # Cost and performance
    estimated_cost: Optional[Decimal] = None
    cache_savings: Optional[Decimal] = None

    # Recommendations
    recommendations: List[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# {self.title}",
            "",
            f"**Run ID:** {self.run_seq}",
            f"**Status:** {self.status.value}",
            f"**Input File:** {self.input_file}",
            f"**Duration:** {self.duration_seconds:.2f} seconds",
            "",
            "## Results",
            f"- **Total Rows:** {self.total_rows:,}",
            f"- **Successfully Cleaned:** {self.cleaned_rows:,} ({self.success_rate:.1%})",
            f"- **Quarantined:** {self.quarantined_rows:,}",
            "",
            "## Fixes Applied",
            "### By Rules",
        ]

        for rule_id, count in self.rules_fixes.items():
            lines.append(f"- {rule_id}: {count:,}")

        if self.llm_fixes:
            lines.extend(["", "### By LLM"])
            for column, count in self.llm_fixes.items():
                lines.append(f"- {column}: {count:,}")

        if self.cache_fixes:
            lines.extend(["", "### From Cache"])
            for column, count in self.cache_fixes.items():
                lines.append(f"- {column}: {count:,}")

        if self.quarantine_reasons:
            lines.extend(["", "## Quarantine Breakdown"])
            for reason, count in self.quarantine_reasons.items():
                lines.append(f"- {reason.value}: {count:,}")

        if self.top_issues:
            lines.extend(["", "## Top Issues Found"])
            for i, issue in enumerate(self.top_issues[:10], 1):
                lines.append(f"{i}. {issue.get('description', 'Unknown issue')}: {issue.get('count', 0):,} occurrences")

        if self.estimated_cost is not None:
            lines.extend(["", "## Cost Analysis"])
            lines.append(f"- **LLM Cost:** ${self.estimated_cost:.4f}")
            if self.cache_savings:
                lines.append(f"- **Cache Savings:** ${self.cache_savings:.4f}")

        if self.recommendations:
            lines.extend(["", "## Recommendations"])
            for rec in self.recommendations:
                lines.append(f"- {rec}")

        return "\n".join(lines)


# =====================================================================
# DIFF CONTRACT
# =====================================================================

class DiffEntry(BaseModel):
    """Entry in the diff file showing what changed."""

    row_number: int
    row_uuid: UUID
    column_name: str
    before_value: Optional[str]
    after_value: str
    source: SourceType
    reason: str
    confidence: float = 1.0  # Default confidence for rules
    timestamp: Optional[datetime] = None  # When the change was made

    def to_csv_dict(self) -> dict:
        """Convert to dict for CSV output."""
        return {
            "row_number": self.row_number,
            "row_uuid": str(self.row_uuid),
            "column_name": self.column_name,
            "before_value": self.before_value or "",
            "after_value": self.after_value,
            "source": self.source.value,
            "reason": self.reason
        }


# =====================================================================
# ERROR/QUARANTINE CONTRACT
# =====================================================================

class QuarantineRow(BaseModel):
    """Row that failed validation and was quarantined."""

    row_uuid: UUID
    row_number: int
    error_category: ErrorCategory
    error_details: List[str]
    original_data: Dict[str, Any]
    attempted_fixes: List[Patch] = Field(default_factory=list)

    def to_csv_dict(self) -> dict:
        """Convert to dict for errors.csv output."""
        result = {
            "row_uuid": str(self.row_uuid),
            "row_number": self.row_number,
            "error_category": self.error_category.value,
            "error_details": "; ".join(self.error_details)
        }
        # Add original data columns
        result.update(self.original_data)
        return result
