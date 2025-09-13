# ADR-001: Rules-First Architecture with Contracted LLM

## Status
Accepted

## Context
We need a data cleaning pipeline that is trustworthy for controllers with AI-skeptical leadership. The system must balance automation with transparency, providing clear audit trails for every transformation while minimizing "black box" AI operations.

## Decision
We will implement a **rules-first architecture** where:
1. Deterministic rules handle 80%+ of data cleaning
2. LLM assistance is tightly scoped to specific enum mappings only
3. Every transformation is tracked with source, reason, and confidence

## Architectural Principles

### 1. Deterministic by Default
- All transformations that can be expressed as rules MUST be rules
- Rules are idempotent - running twice produces no changes
- Identical inputs always produce identical outputs

### 2. Contracted LLM Integration
- LLM only processes two columns: Department and Account Name
- Strict JSON response contract with validation
- Temperature 0 and fixed seed for reproducibility
- Edit caps prevent runaway AI changes (20% max)

### 3. Complete Auditability
- Every cell change generates:
  - Patch record (before/after values)
  - Audit event (timestamp, source, reason)
  - Diff entry (for CSV export)
- Confidence scores for all LLM decisions
- Quarantine for low-confidence or failed mappings

## Implementation Details

### Phase Separation
```
Input → Validation → Rules → Residuals → LLM → Apply → Re-validate → Output
```

Each phase is discrete and can be:
- Tested independently
- Monitored separately
- Rolled back if needed

### Data Flow Principles
1. **Immutability**: Original data never modified
2. **Preconditions**: Patches only apply if current value matches expected
3. **Fail-Safe**: Uncertain data quarantined, not corrupted

### Technology Choices

#### Why Not RAG?
- RAG introduces non-determinism
- Retrieval quality varies with corpus changes
- Harder to audit and explain decisions
- Not needed for simple enum mappings

#### Why LiteLLM?
- Model-agnostic interface
- Easy provider switching
- Built-in retry logic
- Cost tracking support

#### Why Content-Addressed Storage?
- Automatic deduplication
- Immutable artifact references
- Cache-friendly architecture
- Simplified versioning

## Consequences

### Positive
- ✅ High trust due to transparency
- ✅ Predictable behavior and costs
- ✅ Easy to debug and audit
- ✅ Gradual LLM adoption path
- ✅ Clear compliance story

### Negative
- ❌ More complex than pure LLM solution
- ❌ Requires maintaining rule sets
- ❌ Limited to predefined transformations
- ❌ May miss complex patterns

### Neutral
- 🔄 Requires careful schema design
- 🔄 Need to maintain canonical value lists
- 🔄 Performance depends on rule efficiency

## Alternatives Considered

### 1. Pure LLM Approach
**Rejected because:**
- Too much "black box" processing
- Unpredictable costs
- Non-deterministic results
- Hard to audit

### 2. Pure Rules Engine
**Rejected because:**
- Cannot handle variations (abbreviations, typos)
- Requires perfect input data
- No learning capability
- High maintenance burden

### 3. RAG-Enhanced Pipeline
**Rejected because:**
- Adds complexity without clear benefit
- Non-deterministic retrieval
- Requires corpus maintenance
- Overkill for enum mappings

## Metrics for Success

1. **Determinism**: 100% identical outputs for identical inputs
2. **Rules Coverage**: >80% of fixes via rules
3. **LLM Accuracy**: >95% correct mappings
4. **Audit Completeness**: 100% of changes tracked
5. **Performance**: <1 second per 1000 rows for rules

## Security Considerations

1. **No PII in LLM**: Only enum fields sent to AI
2. **Injection Prevention**: All inputs sanitized
3. **Contract Validation**: Strict output checking
4. **Rate Limiting**: Prevent abuse
5. **Secure Storage**: Encrypted artifacts

## Future Evolution

### Phase 1 (Current)
- Two LLM-enabled columns
- Simple enum mappings
- Manual canonical lists

### Phase 2 (Planned)
- Additional columns based on success
- Confidence learning
- Automated canonical list updates

### Phase 3 (Future)
- Semantic understanding for descriptions
- Complex business rule inference
- Human-in-the-loop for uncertainty

## References

- [The Case for Deterministic Data Processing](https://example.com)
- [Audit Requirements for Financial Data](https://example.com)
- [LLM Contract Design Patterns](https://example.com)

## Decision Makers

- Engineering Lead: [Name]
- Product Owner: [Name]
- Compliance Officer: [Name]

## Date
2024-01-15

## Review Date
2024-07-15 (6 months)