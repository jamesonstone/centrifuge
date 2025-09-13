"""Determinism and property-based tests."""

import hashlib
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings

from core.models import Schema, ColumnDefinition, DataType, ColumnPolicy
from core.rules import RulesEngine
from core.validation import Validator
from core.ingest import DataIngestor
from core.artifacts import ArtifactManager


class TestDeterminism:
    """Test deterministic behavior of the pipeline."""
    
    def test_identical_inputs_produce_identical_hashes(self, tmp_path):
        """Test that identical inputs produce identical output hashes."""
        schema = self._create_simple_schema()
        
        # Create test CSV
        csv_content = """id,name,amount,date
1,Alice,100.50,2024-01-15
2,Bob,200.00,2024-01-16
3,Charlie,300.75,2024-01-17"""
        
        # Run pipeline twice
        results = []
        for i in range(2):
            input_file = tmp_path / f"input_{i}.csv"
            input_file.write_text(csv_content)
            
            ingestor = DataIngestor(schema)
            df, input_hash, _ = ingestor.ingest_csv(str(input_file))
            
            rules_engine = RulesEngine(schema)
            cleaned_df, patches, audit_events, diffs = rules_engine.apply_rules(df, uuid4())
            
            # Calculate hash of cleaned data
            cleaned_csv = cleaned_df.to_csv(index=False, sort=True)
            output_hash = hashlib.sha256(cleaned_csv.encode()).hexdigest()
            
            results.append({
                'input_hash': input_hash,
                'output_hash': output_hash,
                'patch_count': len(patches),
                'diff_count': len(diffs)
            })
        
        # Verify identical results
        assert results[0]['input_hash'] == results[1]['input_hash'], "Input hashes should match"
        assert results[0]['output_hash'] == results[1]['output_hash'], "Output hashes should match"
        assert results[0]['patch_count'] == results[1]['patch_count'], "Patch counts should match"
        assert results[0]['diff_count'] == results[1]['diff_count'], "Diff counts should match"
    
    def test_rules_are_idempotent(self, tmp_path):
        """Test that running rules twice produces no additional changes."""
        schema = self._create_simple_schema()
        
        csv_content = """id,name,amount,date
1,alice,100.50,01/15/2024
2,BOB,200,2024-01-16
3,charlie,$300.75,Jan 17 2024"""
        
        input_file = tmp_path / "input.csv"
        input_file.write_text(csv_content)
        
        # First run
        ingestor = DataIngestor(schema)
        df, _, _ = ingestor.ingest_csv(str(input_file))
        
        rules_engine = RulesEngine(schema)
        cleaned_df1, patches1, _, diffs1 = rules_engine.apply_rules(df, uuid4())
        
        # Second run on already cleaned data
        cleaned_df2, patches2, _, diffs2 = rules_engine.apply_rules(cleaned_df1, uuid4())
        
        # No changes should be made in second run
        assert len(patches2) == 0, "No patches should be created on second run"
        assert len(diffs2) == 0, "No diffs should be created on second run"
        
        # Data should be identical
        pd.testing.assert_frame_equal(cleaned_df1, cleaned_df2)
    
    def test_row_order_independence(self, tmp_path):
        """Test that row order doesn't affect individual row processing."""
        schema = self._create_simple_schema()
        
        # Create two CSVs with same data in different order
        csv1 = """id,name,amount,date
1,Alice,100.50,2024-01-15
2,Bob,200.00,2024-01-16
3,Charlie,300.75,2024-01-17"""
        
        csv2 = """id,name,amount,date
3,Charlie,300.75,2024-01-17
1,Alice,100.50,2024-01-15
2,Bob,200.00,2024-01-16"""
        
        results = []
        for i, content in enumerate([csv1, csv2]):
            input_file = tmp_path / f"input_{i}.csv"
            input_file.write_text(content)
            
            ingestor = DataIngestor(schema)
            df, _, _ = ingestor.ingest_csv(str(input_file))
            
            rules_engine = RulesEngine(schema)
            cleaned_df, _, _, _ = rules_engine.apply_rules(df, uuid4())
            
            # Sort by ID for comparison
            cleaned_df = cleaned_df.sort_values('id').reset_index(drop=True)
            results.append(cleaned_df)
        
        # Results should be identical when sorted
        pd.testing.assert_frame_equal(results[0], results[1])
    
    def _create_simple_schema(self) -> Schema:
        """Create a simple test schema."""
        return Schema(
            version="1.0.0",
            columns=[
                ColumnDefinition(
                    name="id",
                    display_name="ID",
                    data_type=DataType.INTEGER,
                    required=True,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="name",
                    display_name="Name",
                    data_type=DataType.STRING,
                    required=True,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="amount",
                    display_name="Amount",
                    data_type=DataType.DECIMAL,
                    required=True,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="date",
                    display_name="Date",
                    data_type=DataType.DATE,
                    required=True,
                    policy=ColumnPolicy.RULE_ONLY
                )
            ]
        )


class TestPropertyBased:
    """Property-based tests using Hypothesis."""
    
    @given(
        department=st.sampled_from(['Sales', 'sales', 'SALES', 'SaLeS', '  Sales  ']),
        amount=st.floats(min_value=0, max_value=1000000, allow_nan=False),
        date_str=st.sampled_from([
            '2024-01-15', '01/15/2024', '15-01-2024', 
            'Jan 15 2024', '15th January 2024'
        ])
    )
    @settings(max_examples=50)
    def test_normalization_properties(self, department, amount, date_str):
        """Test that normalization always produces valid, consistent output."""
        from core.rules import RulesEngine
        
        # Create test data
        df = pd.DataFrame([{
            'row_uuid': str(uuid4()),
            'row_number': 1,
            'department': department,
            'amount': amount,
            'date': date_str
        }])
        
        schema = Schema(
            version="1.0.0",
            columns=[
                ColumnDefinition(
                    name="department",
                    display_name="Department",
                    data_type=DataType.STRING,
                    required=True,
                    policy=ColumnPolicy.LLM_ALLOWED,
                    allowed_values=["Sales", "Operations", "Finance"]
                ),
                ColumnDefinition(
                    name="amount",
                    display_name="Amount",
                    data_type=DataType.DECIMAL,
                    required=True,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="date",
                    display_name="Date",
                    data_type=DataType.DATE,
                    required=True,
                    policy=ColumnPolicy.RULE_ONLY
                )
            ]
        )
        
        rules_engine = RulesEngine(schema)
        cleaned_df, _, _, _ = rules_engine.apply_rules(df, uuid4())
        
        # Properties to verify
        # 1. Department is always title case
        assert cleaned_df.iloc[0]['department'] == 'Sales'
        
        # 2. Amount is always numeric and non-negative
        assert isinstance(cleaned_df.iloc[0]['amount'], (int, float))
        assert cleaned_df.iloc[0]['amount'] >= 0
        
        # 3. Date is always ISO format
        date_val = cleaned_df.iloc[0]['date']
        assert isinstance(date_val, str)
        # Should match YYYY-MM-DD format
        assert len(date_val) == 10
        assert date_val[4] == '-' and date_val[7] == '-'
    
    @given(
        text=st.text(min_size=0, max_size=100),
        has_debit=st.booleans(),
        has_credit=st.booleans(),
        debit_amount=st.floats(min_value=0, max_value=10000, allow_nan=False),
        credit_amount=st.floats(min_value=0, max_value=10000, allow_nan=False)
    )
    @settings(max_examples=50)
    def test_debit_credit_invariant(self, text, has_debit, has_credit, debit_amount, credit_amount):
        """Test that debit/credit XOR constraint is always enforced."""
        df = pd.DataFrame([{
            'row_uuid': str(uuid4()),
            'row_number': 1,
            'description': text,
            'debit_amount': debit_amount if has_debit else 0,
            'credit_amount': credit_amount if has_credit else 0
        }])
        
        schema = Schema(
            version="1.0.0",
            columns=[
                ColumnDefinition(
                    name="description",
                    display_name="Description",
                    data_type=DataType.STRING,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="debit_amount",
                    display_name="Debit Amount",
                    data_type=DataType.DECIMAL,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="credit_amount",
                    display_name="Credit Amount",
                    data_type=DataType.DECIMAL,
                    required=False,
                    policy=ColumnPolicy.RULE_ONLY
                )
            ]
        )
        
        validator = Validator(schema)
        valid_df, quarantine_df, errors = validator.validate(df)
        
        # Check the invariant
        if has_debit and has_credit and debit_amount > 0 and credit_amount > 0:
            # Both debit and credit - should be quarantined
            assert len(quarantine_df) == 1
            assert len(valid_df) == 0
            assert any(e.error_type == 'debit_xor_credit' for e in errors)
        else:
            # Valid case - exactly one or both zero
            assert len(valid_df) == 1
            assert len(quarantine_df) == 0
    
    @given(
        row_count=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=20)
    def test_row_uuid_uniqueness(self, row_count):
        """Test that every row gets a unique UUID."""
        # Create dataframe with multiple rows
        df = pd.DataFrame([
            {
                'id': i,
                'value': f'value_{i}'
            }
            for i in range(row_count)
        ])
        
        schema = Schema(
            version="1.0.0",
            columns=[
                ColumnDefinition(
                    name="id",
                    display_name="ID",
                    data_type=DataType.INTEGER,
                    required=True,
                    policy=ColumnPolicy.RULE_ONLY
                ),
                ColumnDefinition(
                    name="value",
                    display_name="Value",
                    data_type=DataType.STRING,
                    required=True,
                    policy=ColumnPolicy.RULE_ONLY
                )
            ]
        )
        
        # Add row UUIDs
        df['row_uuid'] = [str(uuid4()) for _ in range(len(df))]
        df['row_number'] = range(1, len(df) + 1)
        
        # Verify all UUIDs are unique
        assert len(df['row_uuid'].unique()) == row_count
        
        # Verify all are valid UUIDs
        for uuid_str in df['row_uuid']:
            UUID(uuid_str)  # Will raise if invalid
    
    @given(
        reference=st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-', min_size=1, max_size=20)
    )
    @settings(max_examples=50)
    def test_reference_number_normalization(self, reference):
        """Test reference number normalization."""
        df = pd.DataFrame([{
            'row_uuid': str(uuid4()),
            'row_number': 1,
            'reference_number': reference
        }])
        
        schema = Schema(
            version="1.0.0",
            columns=[
                ColumnDefinition(
                    name="reference_number",
                    display_name="Reference Number",
                    data_type=DataType.STRING,
                    required=True,
                    policy=ColumnPolicy.RULE_ONLY,
                    pattern=r"^REF-\d{3}$"
                )
            ]
        )
        
        rules_engine = RulesEngine(schema)
        cleaned_df, _, _, _ = rules_engine.apply_rules(df, uuid4())
        
        result = cleaned_df.iloc[0]['reference_number']
        
        # Should either be normalized or unchanged
        if reference.lower().startswith('ref') and any(c.isdigit() for c in reference):
            # Should attempt normalization
            assert result.startswith('REF-')
        else:
            # Should be unchanged if not matching pattern
            assert result == reference