# LLM Integration Status Report
Generated: 2025-09-14

## Summary
The Centrifuge project uses Large Language Models (LLMs) for semantic canonicalization of specific data columns that cannot be reliably standardized using deterministic rules alone.

## Current Configuration

### Model Settings
- **Model**: `gpt-4o` (updated from `gpt-4`)
- **Temperature**: `0.0` (deterministic output)
- **Seed**: `42` (reproducible results)
- **Response Format**: JSON mode (when supported)

### Integration Method
- **Library**: LiteLLM (direct library integration)
- **Fallback**: Mock mode for testing/development
- **Retry Logic**: 3 attempts with exponential backoff
- **Caching**: PostgreSQL-based mapping cache

## LLM Usage Scope

### Columns Processed by LLM
1. **Department** 
   - Policy: `LLM_ALLOWED`
   - Canonical values: Sales, Operations, Admin, IT, Finance, Marketing, HR, Legal, Engineering, Support
   - Example: "Tech Support" → "IT"

2. **Account Name**
   - Policy: `LLM_ALLOWED`
   - Canonical values: Cash, Accounts Receivable, Accounts Payable, Sales Revenue, Cost of Goods Sold, Operating Expenses, Equipment, Inventory, Retained Earnings
   - Example: "A/R" → "Accounts Receivable"

### Columns NOT Processed by LLM
All other columns use `RULE_ONLY` policy:
- Transaction ID
- Date
- Account Code
- Amount
- Reference Number
- Created By

## Processing Pipeline

```
1. Ingest & Validation
   ↓
2. Rules Engine (80%+ fixes)
   ↓
3. Residual Planning
   ↓
4. LLM Processing (max 20% edits)
   ↓
5. Final Validation
```

## Recent Updates (2025-09-14)

### Changes Made
1. **Model Upgrade**: Updated from `gpt-4` to `gpt-4o`
   - Enables JSON response format support
   - Better performance and reliability

2. **Best Practices Implementation**:
   - Set temperature to 0.0 for deterministic output
   - Added seed parameter for reproducibility
   - Enabled `litellm.drop_params` for graceful degradation
   - Enhanced error handling for unsupported parameters

3. **Prompt Improvements**:
   - Added explicit JSON-only output instructions
   - Clearer formatting requirements
   - Better examples in prompts

4. **Environment Configuration**:
   - Updated docker-compose.yaml to use `gpt-4o`
   - Support for environment variable overrides
   - Improved configuration flexibility

## Current Issues

### ⚠️ API Key Quota
- **Status**: OpenAI API key has exceeded quota
- **Error**: "You exceeded your current quota, please check your plan and billing details"
- **Impact**: Real LLM calls will fail until quota is restored
- **Workaround**: Use `USE_MOCK_LLM=true` for testing

## Operating Modes

### Production Mode (when API key is valid)
```bash
USE_MOCK_LLM=false
LLM_MODEL_ID=gpt-4o
LLM_TEMPERATURE=0
LLM_SEED=42
```

### Development/Testing Mode
```bash
USE_MOCK_LLM=true
# Uses deterministic mock responses
```

## Architecture Notes

### Current: Direct Library Integration
**Pros:**
- Simpler setup (no proxy service)
- Lower latency
- Direct API calls

**Cons:**
- No centralized logging
- Limited monitoring
- Single provider only

### Alternative: LiteLLM Proxy (commented out)
**Pros:**
- Centralized key management
- Request/response logging
- Cost tracking
- Multi-provider support
- Rate limiting

**Cons:**
- Additional service to manage
- Slightly higher latency

## Recommendations

### Immediate Actions
1. **Fix API Key**: Ensure OpenAI account has valid billing/quota
2. **Use Mock Mode**: For development/testing, set `USE_MOCK_LLM=true`

### Future Improvements
1. **Enable LiteLLM Proxy**: For production deployments
2. **Add Monitoring**: Track LLM usage, costs, and performance
3. **Expand Coverage**: Consider additional columns for LLM processing
4. **Fine-tuning**: Train custom models for domain-specific canonicalization

## Testing

### Verify Mock Mode
```bash
USE_MOCK_LLM=true uv run python -c "
from core.llm_client import LLMClient
import asyncio
async def test():
    client = LLMClient()
    result = await client.process({
        'column': 'Department',
        'values': ['Tech Support'],
        'canonical_values': ['IT', 'Support']
    })
    print(result)
asyncio.run(test())
"
```

### Check Real LLM (when API key is valid)
```bash
USE_MOCK_LLM=false LLM_MODEL_ID=gpt-4o uv run python -c "
from core.llm_client import LLMClient
import asyncio
async def test():
    client = LLMClient()
    print(f'Model: {client.model}')
    print(f'Temperature: {client.temperature}')
asyncio.run(test())
"
```

## Files Modified
- `core/llm_client.py` - Updated model, added best practices
- `docker-compose.yaml` - Changed from gpt-5 to gpt-4o

## Contact
For questions about LLM integration, refer to:
- Architecture Decision Record: `docs/ADR-001-rules-first-llm.md`
- Implementation details: `core/llm_client.py`
- Pipeline integration: `core/pipeline.py`