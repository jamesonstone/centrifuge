
# Alternatives considered and rationale for chosen solution.

version: v1

## purpose

Centrifuge is a proof-of-concept system to clean recurring CSV exports using rules-first transformations with optional LLM assistance. The design emphasizes determinism, auditability, and transparency, addressing the needs of a tech-savvy controller and an AI-skeptical boss.

## key decisions

- **Workers as a pool**: horizontally scalable, each claiming a job (`run_id`) and executing it end-to-end.
- **Orchestrator model**: validate → rules → residual planner → LLM patches → post-validate → emit artifacts.
- **Storage and state**: Postgres holds run metadata; MinIO stores artifacts; API surfaces both.
- **LLM safety**: LiteLLM (via proxy or SDK) with temp=0, JSON schema contracts, patch preconditions, edit caps, and confidence floors.
- **Artifacts**: `cleaned.csv`, `errors.csv`, `diff.csv`, `audit.ndjson`, `manifest.json`, `metrics.json`, `summary.md`.
- **Auditability**: every change logged with before/after, source (rule|llm), reason, and confidence.
- **Skeptic-facing summary**: % by rules vs LLM, quarantine breakdown, “no silent changes.”

## workflow (job lifecycle)

1. **Submit job** via `POST /runs`. API normalizes headers, computes hash, and records metadata in Postgres. State = `queued`.
2. **Worker claim**: one worker atomically marks run as `running`, sets `claimed_by`, and starts heartbeat.
3. **Execution**:
   - Pre-validation (schema checks).
   - Rules-first cleanup.
   - Residual planner identifies violations.
   - LLM invoked only for `llm_allowed` columns; outputs validated patches.
   - Post-validation ensures contract compliance.
   - Non-conforming rows quarantined.
4. **Artifacts** written to MinIO; URIs logged in Postgres.
5. **Completion**: worker updates run state → `succeeded`, `partial`, or `failed`.
6. **Client access**: `GET /runs/{id}` for status/metrics; `GET /runs/{id}/artifacts` for results.

## scaling model

- Start with **2 workers** in docker-compose to show concurrency.
- Pool can scale horizontally without bound.
- Postgres polling (`FOR UPDATE SKIP LOCKED`) is sufficient for PoC.
- In production, replace with **Kafka/Redpanda**: API publishes `run.created`, workers consume as a group, ensuring durability, replay, and backpressure.
- Postgres remains the source of truth for state and artifact pointers.

## why not rag (retrieval-augmented generation)

RAG is **not suitable** here as a primary architecture:

- **Determinism required**: RAG retrieval introduces variability; identical inputs may yield different context windows, breaking reproducibility.
- **Small domain**: the knowledge base (schema, enums, mapping cache) is small and structured. A DB table outperforms vector search for accuracy and simplicity.
- **Auditability**: RAG context is opaque; reviewers can’t easily verify “why” a canonical mapping was chosen. Contracts and caches are explicit and testable.
- **Time constraints**: adding vector infra, retrievers, and eval would consume build time without clear benefit.

RAG can be referenced as a **future enhancement**:

- Use pgvector to store canonical mappings and exemplars.
- Retrieval could supplement LLM prompts with curated few-shots.
- Still requires strict contracts and pinning to preserve reproducibility.

## future enhancements

- Kafka/Redpanda queue for durability and scaling.
- Multi-tenant isolation and per-tenant concurrency caps.
- Canary datasets for new recipes.
- Human-in-the-loop for quarantines.
- Optional retrieval layer for curated exemplars.
- RBAC and auth for API endpoints.

## reviewer-facing narrative

Centrifuge demonstrates:

- **Product sense**: solves a business problem with repeatability and trust.
- **Engineering judgment**: rules-first core, safe LLM integration, audit + artifacts.
- **Scalability**: worker pool model with clear upgrade path (Kafka, more workers).
- **Skeptic accommodation**: every change is traceable, quarantines are explicit, and summaries are readable.

---

## caching and cache misses

Centrifuge uses a **canonical mapping cache** to avoid repeated LLM calls and ensure deterministic outputs:

- **Cache hit**: a messy value (e.g., `"M"`) has already been mapped (`"M" → "Male"`). The worker reuses this mapping.
- **Cache miss**: a messy value has not been seen before. The worker queries the LLM under strict contracts, then writes the result back into the cache for future use.

**Benefits**
- **Cost control**: prevents duplicate LLM queries.
- **Determinism**: guarantees the same messy value always maps to the same canonical output.

---

## infrastructure note: redis (future)

For the PoC, the cache will live in **Postgres** (table `canonical_mapping`), indexed on the messy `variant` string. This avoids extra infrastructure and is sufficient for demonstration.

In production, **Redis** (or another key/value store) could be added in front of Postgres as a **hot cache** for very low-latency lookups. Redis would accelerate frequent duplicate queries, while Postgres remains the source of truth for durability and auditability.

---

## handling of the `description` field

The `Description` column is treated as **informational only** in this proof of concept:

- **Column policy**: set to `rule_only`. No LLM access, no canonicalization, but the raw text is always preserved in artifacts.
- **Quarantine context**: retained in `errors.csv` so a human reviewer can use the description when analyzing invalid rows.
- **Rationale**: free-text fields like `Description` often contain context but are not reliable for deterministic normalization. Using them to infer missing IDs or codes would reduce trust and introduce variance.
- **Future potential**: in later iterations, `Description` could be paired with retrieval or embeddings to help impute missing values or detect semantic anomalies. For the PoC, this is explicitly deferred to maintain determinism, clarity, and auditability.

---

## handling of the `transaction id` field

The `Transaction ID` column is treated as a **business key**, not the sole primary key:

- **Surrogate key**: every row is assigned a unique `row_uuid` (UUIDv7) to serve as the immutable surrogate primary key. This guarantees stable joins across artifacts (`diff.csv`, `audit.ndjson`, `errors.csv`).
- **Validation**: `Transaction ID` is checked for uniqueness and presence.
  - If present and unique across the dataset, it is preserved as a trusted business key.
  - If missing on some rows (while present on the majority) or if duplicates exist, those rows are quarantined with `category=validation_failure`.
- **Audit logging**: all violations are logged explicitly in `audit.ndjson` and `diff.csv`, so there are no silent discards or hidden deduplication.
- **Rationale**: separating the surrogate key (`row_uuid`) from the business key (`Transaction ID`) ensures every row has a stable identifier while preserving the domain-specific semantics of the transaction reference.
- **Future potential**: additional entity-resolution techniques could be applied in production to reconcile duplicate or missing `Transaction IDs`, but this PoC defers such predictive methods in favor of deterministic validation and transparent quarantining.
---

## canonical department mappings

For the proof of concept, we will use a **predefined canonical list of departments** (e.g., *Sales, Operations, Admin, IT, Finance, Marketing, HR*). This approach ensures:

- **Determinism**: every run produces the same mappings, independent of model variance.
- **Auditability**: reviewers can verify that every messy variant was mapped into a known, fixed set.
- **Trust**: the AI-skeptical boss persona requires transparency and predictability. A fixed canonical list provides both.

The LLM is used only in a **narrow assist role**: mapping messy or novel variants into this fixed list (e.g., “ops” → “Operations”). Any value that cannot be mapped confidently is quarantined rather than guessed.

Allowing the model to infer new departments dynamically would weaken reproducibility and risk introducing values outside the approved set. That capability is explicitly **out of scope for this PoC** but can be documented as a future enhancement — for example, an opt-in workflow where administrators review and approve new department values before they are added to the canonical list.
