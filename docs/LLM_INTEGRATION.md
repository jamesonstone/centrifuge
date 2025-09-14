# LLM Integration and Prompt Documentation

## Overview

Centrifuge uses tightly-contracted LLM prompts for specific data cleaning tasks. All LLM interactions are:

- **Deterministic**: Temperature 0, fixed seed
- **Contracted**: Strict JSON response format
- **Bounded**: Edit caps and confidence thresholds
- **Auditable**: Every decision tracked

## Prompt Templates

### Department Mapping

**File**: `prompts/department.yaml`

**Purpose**: Map non-standard department names to canonical values

**Contract**:

```json
{
  "mappings": [
    {
      "original": "string",
      "canonical": "Sales|Operations|Finance|IT|HR|Marketing|Admin|null",
      "confidence": 0.0-1.0,
      "reason": "string"
    }
  ]
}
```

**Canonical Values**:

- Sales
- Operations
- Finance
- IT
- HR
- Marketing
- Admin

**Examples**:

- "sales" → "Sales" (1.0 confidence)
- "Ops" → "Operations" (0.95 confidence)
- "IT Dept" → "IT" (0.98 confidence)
- "Unknown123" → null (0.0 confidence)

### Account Name Mapping

**File**: `prompts/account_name.yaml`

**Purpose**: Standardize account names to canonical chart of accounts

**Contract**:

```json
{
  "mappings": [
    {
      "original": "string",
      "canonical": "string|null",
      "confidence": 0.0-1.0,
      "reason": "string"
    }
  ]
}
```

**Canonical Values**:

- Cash
- Accounts Receivable
- Accounts Payable
- Inventory
- Prepaid Expenses
- Property Plant Equipment
- Accumulated Depreciation
- Notes Payable
- Accrued Expenses
- Retained Earnings
- Revenue
- Cost of Goods Sold
- Operating Expenses
- Interest Expense
- Tax Expense
- Petty Cash

**Examples**:

- "A/R" → "Accounts Receivable" (0.95 confidence)
- "cash on hand" → "Cash" (0.98 confidence)
- "AP" → "Accounts Payable" (0.95 confidence)

## Configuration

### Model Settings

- **Model**: gpt-5 (via LiteLLM proxy)
- **Temperature**: 0.0 (deterministic)
- **Seed**: 42 (reproducible)
- **Max Tokens**: 1000
- **Top P**: 1.0

### Batching

- **Batch Size**: 10-20 unique values per request
- **Retry Strategy**: 3 attempts with exponential backoff
- **Timeout**: 30 seconds per request

### Edit Caps

- **Default**: 20% of rows per column
- **Calculation**: `max_edits = total_rows * 0.20`
- **Enforcement**: Stop processing when cap reached

### Confidence Thresholds

- **Minimum**: 0.80
- **Application**: Only apply mappings >= threshold
- **Quarantine**: Low confidence mappings sent to quarantine

## Prompt Engineering Guidelines

### DO:

- ✅ Provide clear context about the domain
- ✅ Include examples of correct mappings
- ✅ Specify the exact JSON structure required
- ✅ Explain confidence scoring criteria
- ✅ List all canonical values explicitly

### DON'T:

- ❌ Allow free-form text responses
- ❌ Accept mappings without confidence scores
- ❌ Process sensitive data (PII, financial amounts)
- ❌ Allow creative interpretations
- ❌ Accept non-canonical values

## Contract Validation

All LLM responses are validated for:

1. **Structure**: Valid JSON matching schema
2. **Types**: Correct data types for all fields
3. **Values**: Canonical values only (or null)
4. **Confidence**: Between 0.0 and 1.0
5. **Completeness**: All requested mappings present

Failed validations result in:

- Retry with same input (up to 3 times)
- Logging of contract violation
- Fallback to quarantine if persistent failure

## Security Considerations

### Input Sanitization

- Escape special characters in values
- Truncate extremely long strings
- Remove control characters
- Validate UTF-8 encoding

### Injection Prevention

- No dynamic prompt construction
- Fixed template structure
- Parameterized value insertion
- Output validation against whitelist

### Data Protection

- No PII in prompts (names, IDs, etc.)
- No financial amounts
- No dates or timestamps
- Only categorical/enum fields

## Monitoring & Metrics

### Success Metrics

- Contract compliance rate
- Average confidence score
- Cache hit rate
- Retry rate

### Failure Modes

- Contract violations
- Low confidence responses
- Timeout errors
- Rate limit exceeded

### Cost Tracking

- Tokens per request
- Requests per run
- Cost per 1000 rows
- Cache savings

## Examples

### Successful Mapping

```json
// Input
["sales", "OPERATIONS", "Fin", "IT Dept"]

// Output
{
  "mappings": [
    {
      "original": "sales",
      "canonical": "Sales",
      "confidence": 1.0,
      "reason": "Direct case normalization"
    },
    {
      "original": "OPERATIONS",
      "canonical": "Operations",
      "confidence": 1.0,
      "reason": "Direct case normalization"
    },
    {
      "original": "Fin",
      "canonical": "Finance",
      "confidence": 0.92,
      "reason": "Common abbreviation for Finance"
    },
    {
      "original": "IT Dept",
      "canonical": "IT",
      "confidence": 0.98,
      "reason": "Department suffix removal"
    }
  ]
}
```

### Failed Mapping

```json
// Input
["XYZ123", "!!!!", ""]

// Output
{
  "mappings": [
    {
      "original": "XYZ123",
      "canonical": null,
      "confidence": 0.0,
      "reason": "No matching department found"
    },
    {
      "original": "!!!!",
      "canonical": null,
      "confidence": 0.0,
      "reason": "Invalid department name"
    },
    {
      "original": "",
      "canonical": null,
      "confidence": 0.0,
      "reason": "Empty value"
    }
  ]
}
```

## Future Improvements

1. **Semantic Caching**: Cache similar mappings
2. **Confidence Learning**: Adjust thresholds based on accuracy
3. **Multi-Model Ensemble**: Compare outputs from multiple models
4. **Active Learning**: Flag uncertain mappings for human review
5. **Context Enhancement**: Include more domain context in prompts
