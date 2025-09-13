"""Integration test for complete pipeline."""

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from core.models import (
    Schema, ColumnDefinition, DataType, ColumnPolicy,
    RunManifest, RunMetrics, SourceType,
    Patch, AuditEvent, Diff, QuarantineRow, ErrorCategory
)
from core.artifacts import ArtifactManager


def test_artifact_generation(tmp_path):
    """Test complete artifact generation flow."""
    
    # Setup test data
    run_id = uuid4()
    
    # Create schema
    schema = Schema(
        version="1.0.0",
        columns=[
            ColumnDefinition(
                name="transaction_id",
                display_name="Transaction ID",
                data_type=DataType.STRING,
                required=True,
                policy=ColumnPolicy.RULE_ONLY
            ),
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
            )
        ]
    )
    
    # Create cleaned dataframe
    cleaned_df = pd.DataFrame({
        'row_uuid': [str(uuid4()) for _ in range(10)],
        'row_number': list(range(1, 11)),
        'transaction_id': [f'TXN{i:04d}' for i in range(1, 11)],
        'department': ['Sales'] * 5 + ['Finance'] * 5,
        'amount': [100.00 + i * 10 for i in range(10)]
    })
    
    # Create quarantine dataframe
    quarantine_df = pd.DataFrame({
        'row_uuid': [str(uuid4()) for _ in range(2)],
        'row_number': [11, 12],
        'transaction_id': ['TXN0011', 'TXN0012'],
        'department': ['Unknown', 'Marketing'],
        'amount': [200.00, 300.00]
    })
    
    # Create manifest
    manifest = RunManifest(
        run_id=run_id,
        created_at=datetime.now(),
        input_hash="abc123def456",
        schema_version="1.0.0",
        model_version="gpt-5",
        prompt_version="v1",
        recipe_version="v1",
        seed=42,
        options={
            "use_inferred": False,
            "dry_run": False,
            "llm_columns": ["department"]
        },
        versions={
            "python": "3.13",
            "centrifuge": "1.0.0"
        },
        idempotency_key="hash_of_input_and_versions"
    )
    
    # Create metrics
    metrics = RunMetrics(
        run_id=run_id,
        total_rows=12,
        clean_rows=10,
        quarantined_rows=2,
        success_rate=0.833,
        quarantine_rate=0.167,
        
        rules_applied=5,
        rules_fixed_count=8,
        rules_fixed_percentage=0.667,
        
        llm_calls_made=3,
        llm_fixed_count=2,
        llm_fixed_percentage=0.167,
        llm_tokens_used=1500,
        llm_cost_estimate=0.03,
        
        cache_hits=1,
        cache_misses=2,
        cache_hit_rate=0.333,
        
        phase_timings={
            "ingest": 0.5,
            "validation_1": 0.2,
            "rules_engine": 0.3,
            "residual_planner": 0.1,
            "llm_adapter": 2.0,
            "apply_patches": 0.2,
            "validation_2": 0.2,
            "artifacts": 0.5
        },
        total_duration_seconds=4.0,
        
        errors_by_category={"validation_failure": 2},
        errors_by_column={"department": 2},
        
        edit_caps={
            "department": {"limit": 10, "used": 2}
        },
        confidence_distribution={
            "0.8-0.9": 1,
            "0.9-1.0": 1
        }
    )
    
    # Create audit events
    audit_events = [
        AuditEvent(
            event_id=uuid4(),
            run_id=run_id,
            timestamp=datetime.now(),
            event_type="data_transformation",
            phase="rules_engine",
            row_uuid=uuid4(),
            column_name="department",
            before_value="sales",
            after_value="Sales",
            source=SourceType.RULE,
            rule_id="normalize_case",
            reason="Normalized to title case",
            confidence=1.0
        ),
        AuditEvent(
            event_id=uuid4(),
            run_id=run_id,
            timestamp=datetime.now(),
            event_type="data_transformation",
            phase="llm_adapter",
            row_uuid=uuid4(),
            column_name="department",
            before_value="Ops",
            after_value="Operations",
            source=SourceType.LLM,
            contract_id="department_mapping",
            reason="Mapped abbreviation to canonical value",
            confidence=0.95
        )
    ]
    
    # Create diffs
    diffs = [
        Diff(
            row_uuid=uuid4(),
            row_number=1,
            column_name="department",
            before_value="sales",
            after_value="Sales",
            source=SourceType.RULE,
            reason="Normalized to title case",
            confidence=1.0,
            timestamp=datetime.now()
        ),
        Diff(
            row_uuid=uuid4(),
            row_number=2,
            column_name="department",
            before_value="Ops",
            after_value="Operations",
            source=SourceType.LLM,
            reason="Mapped abbreviation",
            confidence=0.95,
            timestamp=datetime.now()
        )
    ]
    
    # Create quarantine rows
    quarantine_rows = [
        QuarantineRow(
            row_uuid=uuid4(),
            row_number=11,
            error_category=ErrorCategory.VALIDATION_FAILURE,
            error_details=["Invalid department: Unknown"],
            original_data={
                "transaction_id": "TXN0011",
                "department": "Unknown",
                "amount": 200.00
            },
            attempted_fixes=[]
        ),
        QuarantineRow(
            row_uuid=uuid4(),
            row_number=12,
            error_category=ErrorCategory.VALIDATION_FAILURE,
            error_details=["Invalid department: Marketing not in allowed values"],
            original_data={
                "transaction_id": "TXN0012",
                "department": "Marketing",
                "amount": 300.00
            },
            attempted_fixes=[
                Patch(
                    row_uuid=uuid4(),
                    column_name="department",
                    before_value="Marketing",
                    after_value="Sales",
                    source=SourceType.LLM,
                    confidence=0.6,
                    reason="Low confidence mapping"
                )
            ]
        )
    ]
    
    # Quarantine stats
    quarantine_stats = {
        'errors_by_category': {'validation_failure': 2},
        'top_error_messages': [
            {'message': 'Invalid department', 'count': 2}
        ]
    }
    
    # Initialize artifact manager
    artifact_manager = ArtifactManager(
        run_id=run_id,
        schema=schema,
        minio_client=None  # Use local storage for test
    )
    
    # Generate all artifacts
    artifacts = artifact_manager.generate_all_artifacts(
        cleaned_df=cleaned_df,
        quarantine_df=quarantine_df,
        manifest=manifest,
        metrics=metrics,
        audit_events=audit_events,
        diffs=diffs,
        quarantine_rows=quarantine_rows,
        quarantine_stats=quarantine_stats
    )
    
    # Verify artifacts were created
    assert 'cleaned.csv' in artifacts
    assert 'errors.csv' in artifacts
    assert 'diff.csv' in artifacts
    assert 'audit.ndjson' in artifacts
    assert 'manifest.json' in artifacts
    assert 'metrics.json' in artifacts
    assert 'summary.md' in artifacts
    
    # Check artifact metadata
    for name, metadata in artifacts.items():
        assert 'content_hash' in metadata
        assert 'size_bytes' in metadata
        assert 'storage_path' in metadata
        assert metadata['size_bytes'] > 0
    
    # Verify local files were created
    artifact_dir = Path(f"artifacts/{run_id}")
    assert artifact_dir.exists()
    
    # Check cleaned.csv content
    cleaned_path = artifact_dir / "cleaned.csv"
    assert cleaned_path.exists()
    cleaned_content = cleaned_path.read_text()
    assert "transaction_id" in cleaned_content
    assert "TXN0001" in cleaned_content
    
    # Check summary.md content
    summary_path = artifact_dir / "summary.md"
    assert summary_path.exists()
    summary_content = summary_path.read_text()
    assert "Centrifuge Run Summary" in summary_content
    assert "Total Rows Processed: 12" in summary_content
    assert "Clean Rows: 10" in summary_content
    assert "Quarantined Rows: 2" in summary_content
    
    # Check manifest.json structure
    manifest_path = artifact_dir / "manifest.json"
    assert manifest_path.exists()
    manifest_data = json.loads(manifest_path.read_text())
    assert manifest_data['model_version'] == "gpt-5"
    assert manifest_data['seed'] == 42
    
    # Check metrics.json structure
    metrics_path = artifact_dir / "metrics.json"
    assert metrics_path.exists()
    metrics_data = json.loads(metrics_path.read_text())
    assert metrics_data['total_rows'] == 12
    assert metrics_data['success_rate'] == 0.833
    assert 'phase_timings' in metrics_data
    
    print(f"✅ All artifacts generated successfully in {artifact_dir}")
    print(f"✅ Total artifacts: {len(artifacts)}")
    print(f"✅ Total size: {sum(a['size_bytes'] for a in artifacts.values())} bytes")
    
    # Clean up
    import shutil
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)


if __name__ == "__main__":
    test_artifact_generation(Path.cwd())