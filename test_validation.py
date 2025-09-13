#!/usr/bin/env python3
"""Test script for validation module."""

from pathlib import Path
from pprint import pprint
from decimal import Decimal

from core.ingest import CSVIngester
from core.validation import Validator
from core.models import Schema, ColumnDefinition, ColumnType, SchemaConstraints

def test_validation():
    """Test CSV validation with sample data."""
    
    # Create schema with all validation rules
    schema = Schema(
        name="accounting_v1",
        version="1.0.0",
        columns={
            "Transaction ID": ColumnDefinition(
                type=ColumnType.STRING, 
                required=True,
                min_length=3,
                max_length=20
            ),
            "Date": ColumnDefinition(
                type=ColumnType.DATE, 
                required=True
            ),
            "Account Code": ColumnDefinition(
                type=ColumnType.STRING, 
                required=True
            ),
            "Account Name": ColumnDefinition(
                type=ColumnType.STRING, 
                required=True, 
                llm_enabled=True
            ),
            "Description": ColumnDefinition(
                type=ColumnType.STRING, 
                required=False
            ),
            "Debit Amount": ColumnDefinition(
                type=ColumnType.DECIMAL, 
                required=False
            ),
            "Credit Amount": ColumnDefinition(
                type=ColumnType.DECIMAL, 
                required=False
            ),
            "Department": ColumnDefinition(
                type=ColumnType.STRING, 
                required=True, 
                llm_enabled=True,
                enum_values=schema.canonical_departments if 'schema' in locals() else None
            ),
            "Reference Number": ColumnDefinition(
                type=ColumnType.STRING, 
                required=False
            ),
            "Created By": ColumnDefinition(
                type=ColumnType.STRING, 
                required=False
            ),
        },
        constraints=SchemaConstraints(
            primary_key="Transaction ID",
            unique=["Transaction ID"],
            debit_xor_credit=True
        )
    )
    
    # Test with sample data
    sample_file = Path("/Users/jamesonstone/go/src/github.com/jamesonstone/centrifuge/sample-data/sample_data.csv")
    
    if not sample_file.exists():
        print(f"Sample file not found: {sample_file}")
        return
    
    print("=" * 80)
    print("TESTING VALIDATION")
    print("=" * 80)
    
    # Step 1: Ingest
    print("\n1. INGESTING CSV...")
    ingester = CSVIngester(schema=schema)
    df = ingester.ingest_file(file_path=sample_file)
    print(f"   ✓ Ingested {len(df)} rows")
    
    # Step 2: Validate
    print("\n2. VALIDATING DATA...")
    validator = Validator(schema=schema)
    valid_df, quarantine_df, errors = validator.validate(df)
    
    print(f"   ✓ Valid rows: {len(valid_df)}")
    print(f"   ✗ Quarantined rows: {len(quarantine_df)}")
    print(f"   ! Total errors: {len(errors)}")
    
    # Get validation summary
    summary = validator.get_validation_summary()
    
    print("\n3. VALIDATION SUMMARY:")
    print(f"   Errors by type:")
    for error_type, count in summary['errors_by_type'].items():
        print(f"     - {error_type}: {count}")
    
    print(f"\n   Errors by column:")
    for column, count in summary['errors_by_column'].items():
        print(f"     - {column}: {count}")
    
    print(f"\n   Error categories:")
    for category, count in summary['error_categories'].items():
        if count > 0:
            print(f"     - {category}: {count}")
    
    # Show some example errors
    if errors:
        print("\n4. EXAMPLE ERRORS (first 10):")
        for error in errors[:10]:
            print(f"   - {error}")
    
    # Show quarantined rows
    if len(quarantine_df) > 0:
        print("\n5. QUARANTINED ROWS (first 5):")
        for _, row in quarantine_df.head().iterrows():
            print(f"   Row {row['row_number']}: Transaction ID = {row.get('Transaction ID', 'N/A')}")
            # Find errors for this row
            row_errors = [e for e in errors if e.row_number == row['row_number']]
            for error in row_errors:
                print(f"     → {error.message}")
    
    # Test specific validation rules
    print("\n6. SPECIFIC VALIDATION TESTS:")
    
    # Test debit_xor_credit
    print("   Testing debit_xor_credit rule:")
    test_cases = [
        (100, None, True, "Valid: debit only"),
        (None, 100, True, "Valid: credit only"),
        (100, 100, False, "Invalid: both positive"),
        (None, None, False, "Invalid: both null"),
        (0, 0, False, "Invalid: both zero"),
        (100, 0, True, "Valid: debit positive, credit zero"),
        ("$1,250.00", None, True, "Valid: formatted debit"),
        ("-500", None, False, "Invalid: negative debit"),
    ]
    
    for debit, credit, expected, desc in test_cases:
        result = validator._validate_debit_xor_credit(debit, credit)
        status = "✓" if result == expected else "✗"
        print(f"     {status} {desc}: debit={debit}, credit={credit} -> {result}")
    
    # Test date parsing
    print("\n   Testing date formats:")
    date_tests = [
        "2024-01-15",
        "01/16/2024",
        "Jan 18 2024",
        "20/01/2024",
        "2024-1-19",
        "01-22-2024",
        "24/01/2024",
        "2024.01.23",
        "29-Jan-2024",
        "Feb 6 2024",
        "2024/02/03",
        "NULL",
        "invalid_date"
    ]
    
    for date_str in date_tests:
        is_valid = validator._is_valid_date(date_str)
        status = "✓" if is_valid else "✗"
        print(f"     {status} '{date_str}'")
    
    # Test amount parsing
    print("\n   Testing amount parsing:")
    amount_tests = [
        ("1500", Decimal("1500")),
        ("$1,250.00", Decimal("1250.00")),
        ("\"3,500.00\"", Decimal("3500.00")),
        ("-500", Decimal("-500")),
        ("(500)", Decimal("-500")),
        ("", None),
        ("NULL", None),
    ]
    
    for amount_str, expected in amount_tests:
        result = validator._parse_amount(amount_str)
        status = "✓" if result == expected else "✗"
        print(f"     {status} '{amount_str}' -> {result}")
    
    print("\n" + "=" * 80)
    print("VALIDATION TEST COMPLETE")
    print("=" * 80)
    
    return valid_df, quarantine_df, errors

if __name__ == "__main__":
    test_validation()