"""Golden tests - validate against expected baseline outputs."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from core.models import Schema, ColumnDefinition, DataType, ColumnPolicy, RunOptions
from core.rules import RuleEngine
from core.validation import Validator
from core.ingest import DataIngestor


# Golden test data
GOLDEN_INPUT_CSV = """Transaction ID,Date,Department,Account Name,Description,Debit Amount,Credit Amount,Reference Number,Created By
TXN001,2024-01-15,sales,cash,Initial deposit,,1000.00,REF-001,john.doe
TXN002,2024-01-16,OPERATIONS,accounts receivable,Customer payment,,500.00,REF-002,jane.smith
TXN003,01/17/2024,Finance,Accounts Payable,Vendor payment,750.00,,REF003,admin
TXN004,2024-01-18,IT,cash,Equipment purchase,1200.00,,REF-004,system
TXN005,Jan 19 2024,hr,PETTY CASH,Office supplies,50.00,,REF005,hr.manager
TXN006,2024/01/20,Sales,Accounts Receivable,Sales invoice,,2000.00,REF-006,sales.rep
TXN007,21st January 2024,marketing,Cash,Marketing expense,300.00,,ref007,marketing.lead
TXN008,2024-01-22,operations,accounts payable,Utility bill,150.00,,REF-008,ops.manager
TXN009,23-01-2024,finance,Cash,Bank fees,25.00,,REF-009,cfo
TXN010,2024-01-24,IT,accounts receivable,Service charge,,400.00,REF010,it.support
"""

GOLDEN_CLEANED_CSV = """row_uuid,row_number,transaction_id,date,department,account_name,description,debit_amount,credit_amount,reference_number,created_by
{uuid},1,TXN001,2024-01-15,Sales,Cash,Initial deposit,0.00,1000.00,REF-001,john.doe
{uuid},2,TXN002,2024-01-16,Operations,Accounts Receivable,Customer payment,0.00,500.00,REF-002,jane.smith
{uuid},3,TXN003,2024-01-17,Finance,Accounts Payable,Vendor payment,750.00,0.00,REF-003,admin
{uuid},4,TXN004,2024-01-18,IT,Cash,Equipment purchase,1200.00,0.00,REF-004,system
{uuid},5,TXN005,2024-01-19,HR,Petty Cash,Office supplies,50.00,0.00,REF-005,hr.manager
{uuid},6,TXN006,2024-01-20,Sales,Accounts Receivable,Sales invoice,0.00,2000.00,REF-006,sales.rep
{uuid},7,TXN007,2024-01-21,Marketing,Cash,Marketing expense,300.00,0.00,REF-007,marketing.lead
{uuid},8,TXN008,2024-01-22,Operations,Accounts Payable,Utility bill,150.00,0.00,REF-008,ops.manager
{uuid},9,TXN009,2024-01-23,Finance,Cash,Bank fees,25.00,0.00,REF-009,cfo
{uuid},10,TXN010,2024-01-24,IT,Accounts Receivable,Service charge,0.00,400.00,REF-010,it.support
"""


def create_test_schema() -> Schema:
    """Create test schema for golden tests."""
    from core.models import SchemaConstraints

    return Schema(
        version="1.0.0",
        columns=[
            ColumnDefinition(
                name="transaction_id",  # Matches header alias
                display_name="Transaction ID",
                data_type=DataType.STRING,
                required=True,
                policy=ColumnPolicy.RULE_ONLY,
                is_primary_key=True
            ),
            ColumnDefinition(
                name="date",  # Matches header alias
                display_name="Date",
                data_type=DataType.DATE,
                required=True,
                policy=ColumnPolicy.RULE_ONLY
            ),
            ColumnDefinition(
                name="department",  # Matches header alias
                display_name="Department",
                data_type=DataType.STRING,
                required=True,
                policy=ColumnPolicy.LLM_ALLOWED,
                allowed_values=["Sales", "Operations", "Finance", "IT", "HR", "Marketing", "Admin"]
            ),
            ColumnDefinition(
                name="account_name",  # Matches header alias
                display_name="Account Name",
                data_type=DataType.STRING,
                required=True,
                policy=ColumnPolicy.LLM_ALLOWED,
                allowed_values=["Cash", "Accounts Receivable", "Accounts Payable", "Petty Cash"]
            ),
            ColumnDefinition(
                name="description",  # Matches header alias
                display_name="Description",
                data_type=DataType.STRING,
                required=False,
                policy=ColumnPolicy.RULE_ONLY
            ),
            ColumnDefinition(
                name="debit_amount",  # Matches header alias
                display_name="Debit Amount",
                data_type=DataType.DECIMAL,
                required=False,
                policy=ColumnPolicy.RULE_ONLY
            ),
            ColumnDefinition(
                name="credit_amount",  # Matches header alias
                display_name="Credit Amount",
                data_type=DataType.DECIMAL,
                required=False,
                policy=ColumnPolicy.RULE_ONLY
            ),
            ColumnDefinition(
                name="reference_number",  # Matches header alias
                display_name="Reference Number",
                data_type=DataType.STRING,
                required=True,
                policy=ColumnPolicy.RULE_ONLY,
                pattern=r"^REF-\d{3}$"
            ),
            ColumnDefinition(
                name="created_by",  # Matches header alias
                display_name="Created By",
                data_type=DataType.STRING,
                required=True,
                policy=ColumnPolicy.RULE_ONLY
            )
        ],
        header_aliases={
            "Transaction ID": "transaction_id",
            "Date": "date",
            "Department": "department",
            "Account Name": "account_name",
            "Description": "description",
            "Debit Amount": "debit_amount",
            "Credit Amount": "credit_amount",
            "Reference Number": "reference_number",
            "Created By": "created_by"
        },
        constraints=SchemaConstraints(debit_xor_credit=True)
    )


class TestGolden:
    """Golden tests for expected outputs."""

    def test_rules_engine_golden(self, tmp_path):
        """Test rules engine produces expected transformations."""
        # Setup
        schema = create_test_schema()

        # Create input CSV
        input_file = tmp_path / "input.csv"
        input_file.write_text(GOLDEN_INPUT_CSV)

        # Ingest data
        ingestor = DataIngestor(schema)
        df, _, _ = ingestor.ingest_csv(str(input_file))

        # Apply rules
        rules_engine = RuleEngine(schema)
        cleaned_df, patches, audit_events, diffs = rules_engine.apply_rules(df, uuid4())

        # Verify transformations
        assert len(cleaned_df) == 10, "Should have 10 rows"

        # Check specific transformations
        # Department normalization
        assert cleaned_df.iloc[0]['department'] == 'Sales'  # was 'sales'
        assert cleaned_df.iloc[1]['department'] == 'Operations'  # was 'OPERATIONS'
        assert cleaned_df.iloc[6]['department'] == 'Marketing'  # was 'marketing'

        # Account name normalization
        assert cleaned_df.iloc[0]['account_name'] == 'Cash'  # was 'cash'
        assert cleaned_df.iloc[1]['account_name'] == 'Accounts Receivable'  # was 'accounts receivable'
        assert cleaned_df.iloc[4]['account_name'] == 'Petty Cash'  # was 'PETTY CASH'

        # Date normalization
        assert cleaned_df.iloc[2]['date'] == '2024-01-17'  # was '01/17/2024'
        assert cleaned_df.iloc[4]['date'] == '2024-01-19'  # was 'Jan 19 2024'
        assert cleaned_df.iloc[6]['date'] == '2024-01-21'  # was '21st January 2024'

        # Reference number normalization
        assert cleaned_df.iloc[2]['reference_number'] == 'REF-003'  # was 'REF003'
        assert cleaned_df.iloc[4]['reference_number'] == 'REF-005'  # was 'REF005'
        assert cleaned_df.iloc[6]['reference_number'] == 'REF-007'  # was 'ref007'

        # Amount normalization
        assert cleaned_df.iloc[0]['debit_amount'] == '0.00'  # was empty
        assert cleaned_df.iloc[0]['credit_amount'] == '1000.00'
        assert cleaned_df.iloc[2]['debit_amount'] == '750.00'
        assert cleaned_df.iloc[2]['credit_amount'] == '0.00'  # was empty

        # Verify patches were created
        assert len(patches) > 0, "Should have patches"

        # Verify audit events
        assert len(audit_events) > 0, "Should have audit events"

        # Verify diffs
        assert len(diffs) > 0, "Should have diffs"
        department_diffs = [d for d in diffs if d.column_name == 'department']
        assert len(department_diffs) >= 4, "Should have department normalizations"

    def test_validation_golden(self):
        """Test validation catches expected errors."""
        schema = create_test_schema()
        validator = Validator(schema)

        # Create test data with known issues
        invalid_data = pd.DataFrame([
            {
                'row_uuid': str(uuid4()),
                'row_number': 1,
                'transaction_id': 'TXN001',
                'date': '2024-01-15',
                'department': 'InvalidDept',  # Not in allowed values
                'account_name': 'Cash',
                'description': 'Test',
                'debit_amount': 100.00,
                'credit_amount': 100.00,  # Both debit and credit (invalid)
                'reference_number': 'INVALID',  # Wrong format
                'created_by': 'test'
            },
            {
                'row_uuid': str(uuid4()),
                'row_number': 2,
                'transaction_id': '',  # Missing required field
                'date': 'invalid-date',  # Invalid date format
                'department': 'Sales',
                'account_name': '',  # Missing required field
                'description': 'Test',
                'debit_amount': -50.00,  # Negative amount
                'credit_amount': 0.00,
                'reference_number': 'REF-001',
                'created_by': ''  # Missing required field
            }
        ])

        valid_df, quarantine_df, errors = validator.validate(invalid_data)

        # Verify both rows are quarantined
        assert len(quarantine_df) == 2, "Both rows should be quarantined"
        assert len(valid_df) == 0, "No rows should be valid"

        # Verify specific errors are caught
        error_types = {e.error_type for e in errors}
        assert 'enum_violation' in error_types, "Should catch invalid enum"
        assert 'debit_xor_credit' in error_types, "Should catch debit/credit violation"
        assert 'pattern_violation' in error_types, "Should catch pattern mismatch"
        assert 'required_field' in error_types, "Should catch missing required fields"

    def test_end_to_end_golden(self, tmp_path):
        """Test complete pipeline produces expected results."""
        from core.artifacts import ArtifactManager
        from core.models import RunManifest, RunMetrics

        # Setup
        schema = create_test_schema()
        run_id = uuid4()

        # Create input CSV
        input_file = tmp_path / "input.csv"
        input_file.write_text(GOLDEN_INPUT_CSV)

        # Phase 1: Ingest
        ingestor = DataIngestor(schema)
        df, input_hash, header_mapping = ingestor.ingest_csv(str(input_file))

        # Phase 2: Initial validation
        validator = Validator(schema)
        _, _, initial_errors = validator.validate(df)

        # Phase 3: Rules engine
        rules_engine = RuleEngine(schema)
        cleaned_df, patches, audit_events, diffs = rules_engine.apply_rules(df, run_id)

        # Phase 4: Final validation
        valid_df, quarantine_df, final_errors = validator.validate(cleaned_df)

        # Phase 5: Generate artifacts
        manifest = RunManifest(
            run_id=run_id,
            run_seq=1,  # Added required field
            input_file_name="input.csv",  # Added required field
            input_file_hash=input_hash,  # Corrected field name
            input_row_count=len(df),  # Added required field
            schema_version=schema.version,
            model_id="gpt-5",  # Corrected field name
            prompt_version="v1",
            engine_version="v1",  # Corrected field name
            seed=42,
            options=RunOptions(),  # Use proper RunOptions object
            started_at=datetime.now(),  # Added required field
            idempotency_key=f"{input_hash}_{schema.version}"
        )

        metrics = RunMetrics(
            run_id=run_id,
            total_rows=len(df),
            total_cells=len(df) * len(df.columns),
            cells_validated=len(df) * len(df.columns),
            cells_modified=len(patches),
            cells_quarantined=len(quarantine_df) * len(df.columns) if len(quarantine_df) > 0 else 0,
            rules_fixed_count=len(patches),
            rules_duration_ms=100,
            llm_duration_ms=0,
            validation_duration_ms=50,
            total_duration_ms=150
        )

        artifact_manager = ArtifactManager(run_id, schema)
        artifacts = artifact_manager.generate_all_artifacts(
            cleaned_df=valid_df,
            quarantine_df=quarantine_df,
            manifest=manifest,
            metrics=metrics,
            audit_events=audit_events,
            diffs=diffs,
            quarantine_rows=[],
            quarantine_stats={}
        )

        # Verify artifacts
        assert 'cleaned.csv' in artifacts
        assert 'manifest.json' in artifacts
        assert 'metrics.json' in artifacts
        assert 'summary.md' in artifacts

        # All rows should be valid after rules
        assert len(valid_df) == 10, "All rows should be valid after rules"
        assert len(quarantine_df) == 0, "No rows should be quarantined"

        print(f"✅ Golden test passed: {len(valid_df)} rows cleaned successfully")
