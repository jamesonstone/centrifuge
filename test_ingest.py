#!/usr/bin/env python3
"""Test script for CSV ingestion."""

import json
from pathlib import Path
from pprint import pprint

from core.ingest import CSVIngester, HeaderMapper
from core.models import Schema, ColumnDefinition, ColumnType, SchemaConstraints

def test_ingestion():
    """Test CSV ingestion with sample data."""
    
    # Create a basic schema
    schema = Schema(
        name="accounting_v1",
        version="1.0.0",
        columns={
            "Transaction ID": ColumnDefinition(type=ColumnType.STRING, required=True),
            "Date": ColumnDefinition(type=ColumnType.DATE, required=True),
            "Account Code": ColumnDefinition(type=ColumnType.STRING, required=True),
            "Account Name": ColumnDefinition(type=ColumnType.STRING, required=True, llm_enabled=True),
            "Description": ColumnDefinition(type=ColumnType.STRING, required=False),
            "Debit Amount": ColumnDefinition(type=ColumnType.DECIMAL, required=False),
            "Credit Amount": ColumnDefinition(type=ColumnType.DECIMAL, required=False),
            "Department": ColumnDefinition(type=ColumnType.STRING, required=True, llm_enabled=True),
            "Reference Number": ColumnDefinition(type=ColumnType.STRING, required=False),
            "Created By": ColumnDefinition(type=ColumnType.STRING, required=False),
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
    print("TESTING CSV INGESTION")
    print("=" * 80)
    
    # Create ingester
    ingester = CSVIngester(schema=schema)
    
    # Ingest file
    print(f"\nIngesting file: {sample_file}")
    df = ingester.ingest_file(file_path=sample_file)
    
    print(f"\n✓ File ingested successfully!")
    print(f"  - Rows: {ingester.row_count}")
    print(f"  - Delimiter: {repr(ingester.delimiter)}")
    print(f"  - Encoding: {ingester.encoding}")
    print(f"  - File hash: {ingester.file_hash[:16]}...")
    
    print(f"\n✓ Headers normalized:")
    for orig, norm in zip(ingester.headers[:5], ingester.normalized_headers[:5]):
        print(f"  - '{orig}' → '{norm}'")
    
    # Profile data
    print(f"\n✓ Profiling data...")
    profile = ingester.profile_data(df)
    
    print(f"\nData Profile:")
    print(f"  - Total cells: {profile['row_count'] * profile['column_count']}")
    print(f"  - Missing values by column:")
    for col, missing in profile['missing_values'].items():
        if missing > 0:
            pct = (missing / profile['row_count']) * 100
            print(f"    - {col}: {missing} ({pct:.1f}%)")
    
    print(f"\n  - Detected data types:")
    for col, dtype in profile['data_types'].items():
        print(f"    - {col}: {dtype}")
    
    print(f"\n  - Unique value counts:")
    for col, unique in profile['unique_counts'].items():
        print(f"    - {col}: {unique}")
    
    # Show sample of data
    print(f"\n✓ Sample data (first 5 rows):")
    print(df[ingester.normalized_headers[:5]].head())
    
    # Test header mapping
    print(f"\n✓ Testing header mapper:")
    test_headers = [
        "txn_id",
        "trans_date", 
        "acct_code",
        "dept",
        "debit_amt",
        "ref_no"
    ]
    for header in test_headers:
        mapped = HeaderMapper.map_header(header)
        print(f"  - '{header}' → '{mapped}'")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_ingestion()