# LLM Operation Flow in Centrifuge Pipeline

## Overview

This document details exactly where and how the LLM operates within the Centrifuge pipeline, with enhanced logging to track its activities.

## Pipeline Phases and LLM Involvement

```plaintext
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE EXECUTION FLOW                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Phase 1: INGEST & VALIDATION                                │
│  └─ ❌ No LLM involvement                                    │
│                                                               │
│  Phase 2: RULES ENGINE                                       │
│  └─ ❌ No LLM involvement (deterministic only)              │
│                                                               │
│  Phase 3: RESIDUAL PLANNING                                  │
│  └─ ⚠️  LLM columns identified but not processed            │
│      • Identifies Department & Account Name columns          │
│      • Counts unique values needing canonicalization         │
│      • Checks cache for existing mappings                    │
│                                                               │
│  Phase 4: LLM PROCESSING  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  └─ ✅ PRIMARY LLM OPERATIONS HERE                          │
│      • Processes residual items from Phase 3                 │
│      • Makes API calls to OpenAI/Mock LLM                    │
│      • Applies canonical mappings                            │
│      • Updates cache with successful mappings                │
│                                                               │
│  Phase 5: FINAL VALIDATION                                   │
│  └─ ❌ No LLM involvement                                    │
│                                                               │
│  Phase 6: ARTIFACT GENERATION                                │
│  └─ ❌ No LLM involvement                                    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Detailed LLM Operation Points

### 1. Column Identification (Phase 3)

**Location**: `core/pipeline.py::plan_residuals()`

The system identifies which columns are eligible for LLM processing:

- **Department**: `ColumnPolicy.LLM_ALLOWED`
- **Account Name**: `ColumnPolicy.LLM_ALLOWED`

All other columns have `ColumnPolicy.RULE_ONLY` and bypass LLM entirely.

### 2. Residual Item Collection (Phase 3)

**Location**: `core/residual_planner.py::identify_residuals()`

For LLM-eligible columns, the system:

- Collects unique values that couldn't be fixed by rules
- Applies edit cap (max 20% of rows can be edited) -- this is purposefully conservative for the PoC but can (and should) be expanded in production
- Checks cache for previously mapped values -- this is just the postgres datastore but could be replaced with Redis or similar for speed at scale

### 3. LLM Processing (Phase 4)

**Location**: `core/pipeline.py::process_llm()` → `core/llm_client.py::process()`

The actual LLM operations occur here:

#### 3.1 Request Preparation

```python
request = {
    'column': 'Department',
    'values': ['Tech Support', 'IT Dept', ...],
    'canonical_values': ['IT', 'Sales', 'Finance', ...]
}
```

#### 3.2 LLM API Call

- Uses LiteLLM library for OpenAI integration
- Model: `gpt-4o` (supports JSON response format)
- Temperature: `0.0` (deterministic)
- Seed: `42` (reproducible)

#### 3.3 Response Processing

```python
response = {
    'success': True,
    'mappings': {
        'Tech Support': 'IT',
        'IT Dept': 'IT',
        'Unknown Dept': None  # Can't map with confidence
    },
    'confidence': 0.85,
    'model': 'gpt-4o'
}
```

#### 3.4 Data Updates

- Applies mappings to DataFrame
- Records changes in audit trail
- Stores successful mappings in cache

## Enhanced Logging Output

With the new logging enhancements, you'll see:

```plaintext
============================================================
LLM PROCESSING PHASE STARTING
============================================================
LLM Adapter Configuration:
  Model: gpt-4o
  Temperature: 0.0
  Seed: 42
  Mock Mode: False

LLM Processing Overview:
  Total patches to process: 2
  Total unique values: 12
  Column 'Department':
    Unique values to canonicalize: 5
    Values: ['Tech Support', 'IT Dept', 'Sales Team', ...]
  Column 'Account Name':
    Unique values to canonicalize: 7
    Sample values: ['A/R', 'Accts Payable', ...]

------------------------------------------------------------
Processing Patch 1/2
  Column: 'Department'
  Values to process: 5
  Calling LLM...
  LLM Response:
    Success: True
    Model: gpt-4o
    Mappings received: 5
    Confidence: 0.85
    Non-null mappings: 4
    Sample mappings: {'Tech Support': 'IT', 'IT Dept': 'IT'}
  Applied 8 fixes to data

Processing Patch 2/2
  Column: 'Account Name'
  Values to process: 7
  Calling LLM...
  LLM Response:
    Success: True
    Model: gpt-4o
    Mappings received: 7
    Confidence: 0.85
    Non-null mappings: 6
    Sample mappings: {'A/R': 'Accounts Receivable', ...}
  Applied 11 fixes to data

------------------------------------------------------------
LLM PROCESSING PHASE COMPLETE
  Total values attempted: 12
  Total values fixed: 19
  Total values errored: 0
  Success rate: 100.0%
============================================================
```

## Log Locations

### Worker Logs

```bash
# View worker logs with LLM details
docker-compose logs -f centrifuge-worker-1 centrifuge-worker-2

# Filter for LLM-specific logs
docker-compose logs centrifuge-worker-1 | grep -E "LLM|llm"
```

### Log Levels

- **INFO**: High-level LLM operations (phase start/end, totals)
- **DEBUG**: Detailed operations (individual mappings, API calls)
- **WARNING**: Failed attempts, retries
- **ERROR**: Critical failures, quota issues

## Monitoring LLM Operations

### Key Metrics to Track

1. **Residual Count**: How many values need LLM processing
2. **Cache Hit Rate**: How often we avoid API calls
3. **Success Rate**: Percentage of successful mappings
4. **Edit Cap Usage**: How close to 20% limit
5. **API Errors**: Rate limits, quota issues

### Sample Log Analysis

```bash
# Count LLM operations per run
docker-compose logs | grep "LLM PROCESSING PHASE COMPLETE" | tail -10

# Check cache effectiveness
docker-compose logs | grep "cache hits" | tail -10

# Monitor API errors
docker-compose logs | grep -E "quota|rate limit" | tail -10
```

## Troubleshooting

### No LLM Activity Visible

Check if:

1. Residual items exist: Look for "Identified X residual items"
2. LLM columns have data: Department and Account Name columns
3. Edit cap not exceeded: Max 20% of rows

### LLM Not Fixing Values

Check if:

1. API key is valid: `OPENAI_API_KEY` in .env
2. Mock mode enabled: `USE_MOCK_LLM=true` bypasses real API
3. Model compatibility: Using `gpt-4o` for JSON support

### Performance Issues

Monitor:

1. Batch sizes: Large value sets may timeout
2. Retry attempts: Multiple failures indicate API issues
3. Cache misses: High miss rate increases API calls

## Configuration

### Environment Variables

```bash
# Core LLM settings
LLM_MODEL_ID=gpt-4o          # Model to use
LLM_TEMPERATURE=0            # Deterministic output
LLM_SEED=42                  # Reproducible results
USE_MOCK_LLM=false           # Use real API

# Performance tuning
EDIT_CAP_PCT=20              # Max % of rows to edit
CONFIDENCE_FLOOR=0.80        # Minimum confidence
```

### Files Involved

- `core/llm_client.py` - LLM API integration
- `core/llm_processor.py` - Enhanced processing with logging
- `core/pipeline.py` - Pipeline orchestration
- `core/residual_planner.py` - Identifies LLM candidates

## Summary

The LLM operates in a very focused manner within Centrifuge:
1. **Only Phase 4** actively uses LLM
2. **Only 2 columns** are processed (Department, Account Name)
3. **Only residuals** that rules couldn't fix
4. **Maximum 20%** of rows can be edited
5. **Caching** prevents repeated API calls

With enhanced logging, every LLM operation is now visible and traceable through the pipeline execution.
