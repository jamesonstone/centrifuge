# Centrifuge — Prompt Foundry Work Order (End-to-End Build Spec)
version: v1

## Phase Table
| # | Phase | Outcome | Progress | Confidence |
|---|------|---------|----------|------------|
| 0 | Repo & Infra Bootstrap | Local stack up via `docker-compose`; health endpoints respond | 100% | 98% |
| 1 | Data Contracts | Finalize JSON contracts for schema, manifest, patch, audit, metrics, summary | 100% | 98% |
| 2 | Ingest & Profiling | Load CSV, sniff delimiter/encoding, normalize headers, compute input hash | 100% | 98% |
| 3 | Validation-1 | Enforce required columns, types, enums, PK, domain rules (`debit_xor_credit`) | 100% | 98% |
| 4 | Rules Engine | Idempotent transforms: alias→canonical, trim/case, typecast, ISO date, enum maps | 100% | 98% |
| 5 | Residual Planner & Cache | Identify violations; batch unique values; lookup/write-through canonical mapping cache | 100% | 98% |
| 6 | LLM Adapter (LiteLLM→gpt-5) | Contract-safe patches for `Department` and `Account Name` | 100% | 98% |
| 7 | Apply & Audit | Apply patches with preconditions; emit per-change audit and diff | 100% | 98% |
| 8 | Validation-2 & Quarantine | Re-validate; categorize failures; write `errors.csv` | 0% | 98% |
| 9 | Artifacts & Summary | Emit cleaned, errors, diff, audit, manifest, metrics, summary; index in DB | 0% | 98% |
|10| API Surface | `POST /runs`, `GET /runs/{id}`, `GET /runs/{id}/artifacts` with OpenAPI | 0% | 98% |
|11| Tests | Golden, determinism, adversarial, property tests green in CI | 0% | 98% |
|12| Docs & Ops Notes | README, prompts.md, ADR, runbook; reviewer can reproduce end-to-end | 0% | 98% |
|13| Experimental Flags | `use_inferred=true`, `allow_in_prompt` plumbing; default off | 0% | 98% |

---

## Objective
Deliver a trustworthy CSV cleaning PoC for a controller with an AI-skeptical boss. Prioritize determinism, auditability, and transparent outputs. Use rules-first transforms with tightly contracted LLM assists on specific columns only.

---

## Locked Business Rules & Constraints
- **Tech stack:** Python 3.13; FastAPI; pandas; JSON-schema style validation with light custom validators; MinIO for artifacts; Postgres for metadata and canonical cache; LiteLLM proxy to **gpt-5**; `uv` package manager.
- **LLM columns (PoC):** `Department`, `Account Name` only. All other columns are `rule_only`.
- **PII exposure control:** `allow_in_prompt=false` for `Transaction ID`, `Account Code`, `Debit Amount`, `Credit Amount`, `Reference Number`, `Date`, `Description`, `Created By`. Only `Department` and `Account Name` are sent to LLM.
- **Keys:** Generate immutable `row_uuid` per row as surrogate PK. Treat `Transaction ID` as a business key; enforce uniqueness if present on the majority. Missing or duplicates → quarantine.
- **Dates:** Canonical `YYYY-MM-DD`.
- **Text canonical form:** Title Case for enums/categories.
- **Row cap:** 50k default.
- **Schema inference:** available but **experimental**; requires explicit `use_inferred=true` and implemented last.
- **Auth:** Pass-through middleware hooks implemented; no auth logic for PoC; document future JWT/API key.
- **Department canonicalization:** Predefined list (Sales, Operations, Admin, IT, Finance, Marketing, HR, etc.); LLM maps only non-matching variants.
- **Account Name canonicalization:** Predefined list for common accounts (Cash, Accounts Receivable, Accounts Payable, etc.); LLM maps variants; unknowns → quarantine.
- **LLM batching:** Chunked batches of 10-20 unique values per prompt; per-value retries within chunks.
- **Canonical cache:** Persistent across runs, versioned with model_id, prompt_version, source (rule|llm|human); prefer latest approved entry.
- **Database keys:** UUIDv7 primary keys everywhere; run_seq for human-friendly display; compound unique index for cache.
- **Run states:** Simple state machine: `queued` → `running` → {`succeeded` | `partial` | `failed`}; sub-phase progress in JSON field.
- **LLM config:** Fail fast on missing OPENAI_API_KEY in prod; allow mock adapter in dev mode; rate limiting configurable.

---

## Global Policies
- **Determinism:** temperature 0; fixed seed 42; version pinning (model, prompt, schema, recipe); idempotency key = hash(input bytes + versions).
- **Patch Preconditions:** each LLM patch specifies `before_value`; apply only if current cell equals `before_value`.
- **Edit Caps & Confidence:** default 20% max edited rows per LLM column; minimum confidence 0.80; below floor → quarantine.
- **Error Categories:** `validation_failure`, `llm_contract_failure`, `low_confidence`, `edit_cap_exceeded`, `parse_error`.
- **Quarantine:** Attempt fixes first (rules → LLM → post-validate); quarantine only if still non-compliant with clear category and reasons.
- **Amount normalization:** Strip formatting (commas, quotes, $); ensure non-negative debit/credit; flip sign if unambiguous; quarantine if ambiguous.
- **Artifact storage:** Content-addressed `/artifacts/{content_hash}/{artifact_name}`; deduped with run_id links in Postgres.
- **Prompt management:** Externalized to `prompts/` as YAML/JSON; versioned independently; prompt_version in manifest.
- **Logging:** Structured with correlation IDs (run_id, worker_id) from start; metrics to run file; Prometheus/StatsD noted as future.

---

## Phase Details

### Phase 0 — Repo & Infra Bootstrap
**Outcome:** Local environment starts with API, two workers, Postgres, MinIO, LiteLLM proxy.
**Implementation Guidance:**
- Define environment variables for DB and MinIO; store no secrets in VCS.
- Database init via raw SQL in `ops/sql/init/` (auto-run by docker-entrypoint); note Alembic for production.
- Health check `/healthz` returns 200 with fast DB ping (`SELECT 1`); skip MinIO/LiteLLM checks for PoC.
- One image, two entrypoints: API and worker share the same codebase.
**Exit Criteria:** `docker-compose up` results in healthy services; API responds; MinIO bucket for artifacts exists.

---

### Phase 1 — Data Contracts
**Outcome:** Stable JSON definitions for: schema, run manifest, LLM patch, audit event, metrics, summary.
**Implementation Guidance:**
- **Schema** includes columns (types, nullability, enums, regex, `allow_in_prompt`, policy), constraints (PK, unique), header aliases, and domain rules.
- **Manifest** pins versions, input hash, options, seed; capture model `gpt-5` and `prompt_version`.
- **Patch** structure contains row identifier, column, before/after, reason, confidence, contract id.
- **Audit** event captures source (`rule|llm`), rule/contract ids, timestamp, and optional confidence.
- **Metrics** cover counts, percentages, latencies, LLM calls, cache hits, token/cost estimate if available.
- **Summary** is a human-readable report file enumerating fixes, quarantines, and costs.
**Exit Criteria:** All contract examples validate; reviewers can understand every field without code.

---

### Phase 2 — Ingest & Profiling
**Outcome:** Robust CSV intake with consistent header normalization and stable input hashing.
**Implementation Guidance:**
- Accept **multipart upload** and **`s3_url`**; stream large files.
- Sniff delimiter and encoding; normalize headers using schema aliases.
- Compute a content hash over raw bytes; store in manifest.
- If no schema provided, stop unless `use_inferred=true` (deferred to Phase 13).
**Exit Criteria:** Sample files ingest; normalized header set derived; stable hash computed.

---

### Phase 3 — Validation-1
**Outcome:** Deterministic detection of problems before any transformation.
**Implementation Guidance:**
- Check required columns present.
- Validate types, enums, regex, and date format candidates.
- Enforce `debit_xor_credit`: exactly one of debit/credit positive; the other null/zero.
- Assign `row_uuid` for every row.
- Assess `Transaction ID`: if majority of rows have it, enforce uniqueness; duplicates/missing → mark violations for later quarantine.
**Exit Criteria:** Violations enumerated without mutating data; reports are stable across runs.

---

### Phase 4 — Rules Engine
**Outcome:** Idempotent, deterministic cleanup that handles the majority of issues.
**Implementation Guidance:**
- Apply header alias→canonical mapping.
- Trim whitespace; collapse internal whitespace where applicable; Title Case for categorical text.
- Typecast numeric fields; non-castable values recorded as violations.
- Normalize dates to ISO; invalid dates flagged.
- Apply enum maps where a canonical list exists.
- No silent deduplication for `Transaction ID`.
**Exit Criteria:** Re-running rules on already-clean output yields no changes.

---

### Phase 5 — Residual Planner & Cache
**Outcome:** Efficient, bounded set of items for LLM remediation with cache fast-path.
**Implementation Guidance:**
- Identify residual violations **only** for `Department` and `Account Name`.
- Build batches by **unique values** rather than per row.
- Look up canonical mapping cache in Postgres first; if hit, apply and log as `rule` source with `source=rule` or `source=cache` as preferred nomenclature.
- Track planned LLM calls and enforce edit caps per column.
**Exit Criteria:** Planner reports unique candidate values, cache hits applied, caps computed.

---

### Phase 6 — LLM Adapter (LiteLLM→gpt-5)
**Outcome:** Contract-safe patches produced for remaining unique values.
**Implementation Guidance:**
- Use LiteLLM proxy with temperature 0 and fixed seed.
- Prompt with column context, allowed values (when enumerated), and examples sourced from schema or curated seed lists.
- Require strict JSON structure; validate shape and types.
- On invalid output, perform bounded retry; on persistent failure, mark `llm_contract_failure`.
- Record confidence; filter below the floor; do not apply low-confidence patches.
**Exit Criteria:** Patches ready for application; retries bounded; failures categorized.

---

### Phase 7 — Apply & Audit
**Outcome:** Patches applied safely; per-change logs generated.
**Implementation Guidance:**
- Apply patches only if `before_value` equals the current cell; otherwise discard with reason.
- Emit one audit event per change including source (`llm`), contract id, reason, confidence.
- Build `diff` entries for both rules-based and LLM-based edits; include the source and reason fields.
**Exit Criteria:** Every mutation is reflected in audit and diff; no blind edits occur.

---

### Phase 8 — Validation-2 & Quarantine
**Outcome:** Final compliance check; unsafe rows isolated for review.
**Implementation Guidance:**
- Re-run the same validations as Phase 3.
- For rows still failing any rule, move to quarantine with a single primary category from the allowed set; collect detailed reasons.
- Ensure quarantined rows keep all original context, including `Description`.
**Exit Criteria:** Clean partition and `errors.csv` created; run outcome determined (`succeeded` vs `partial`).

---

### Phase 9 — Artifacts & Summary
**Outcome:** Complete artifact set persisted and indexed.
**Implementation Guidance:**
- Emit: `cleaned.csv`, `errors.csv`, `diff.csv`, `audit.ndjson`, `manifest.json`, `metrics.json`, `summary.md`.
- Store in MinIO under content-addressed paths; write URIs and hashes to Postgres.
- `summary.md` includes counts, % fixed by rules vs LLM, quarantine breakdown by category, cache hit ratio, per-phase timings, token/cost estimate if available.
**Exit Criteria:** All artifact links resolvable; hashes stable across re-runs.

---

### Phase 10 — API Surface
**Outcome:** Minimal, durable endpoints for submission, status, and artifact access.
**Implementation Guidance:**
- `POST /runs` accepts multipart file and/or `s3_url`, plus options (`use_inferred`, `dry_run`, `llm_columns`).
- `GET /runs/{id}` returns state, counts, quarantine breakdown, timings, versions, and artifact links.
- `GET /runs/{id}/artifacts` enumerates or streams artifacts.
- No auth in PoC; document future auth in README.
**Exit Criteria:** OpenAPI renders; e2e flow from submit→status→download works.

---

### Phase 11 — Tests
**Outcome:** Automated confidence in correctness and reproducibility.
**Implementation Guidance:**
- **Golden**: given a fixture CSV from sample_data.csv and schema, `cleaned.csv` and `diff.csv` match expected baselines.
- **Synthetic**: edge cases for formats, duplicates, injection, boundary conditions.
- **Determinism**: two runs with identical inputs produce identical artifact hashes.
- **Adversarial**: free-text value tries to break the contract; ensure rejection and quarantine.
- **Property**: PK uniqueness preserved; enums closed; dates ISO; `debit_xor_credit` invariant holds.
**Exit Criteria:** All tests pass locally and in CI.

---

### Phase 12 — Docs & Ops Notes
**Outcome:** Clear artifacts for reviewers and operators.
**Implementation Guidance:**
- **README** with quickstart, environment setup (`uv`), compose usage, API examples, and production considerations (auth, tenancy, event bus, canaries, budgets).
- **Makefile** with standard targets: `setup`, `run`, `test`, `lint`, `migrate`, `clean`, `compose-up`, `compose-down`.
- **prompts.md** documenting LLM contracts, examples, edit caps, confidence thresholds, and failure modes.
- **ADR** explaining rules-first + contracted LLM, and why no RAG for the PoC.
- **Runbook** notes for common failures, DLQ approach (future), and data retention.
**Exit Criteria:** A reviewer can clone, run, and evaluate in one pass.

---

### Phase 13 — Experimental Flags
**Outcome:** Optional features available behind explicit opt-in.
**Implementation Guidance:**
- `use_inferred=true` enables light schema inference; require explicit flag and mark output manifest as `schema_version: inferred`.
- `allow_in_prompt` honored per column; defaults off except `Department` and `Account Name`.
- Document both as experimental and non-default; implement last.
**Exit Criteria:** Flags present, off by default, and safe when enabled.

---

## Role Split & Scaling Model
- **API**: thin, stateless submission and read endpoints; no long work.
- **Workers**: pool that claims one run at a time; executes ALL phases end-to-end; writes status, audit, and artifacts.
- **Queueing**: DB-backed claim (`FOR UPDATE SKIP LOCKED`) for PoC; no phase sharding; Kafka/Redpanda noted as future enhancement.
- **Recovery**: heartbeat and visibility timeout allow safe re-claim; idempotency key prevents duplicate effects.
- **Worker execution**: Single worker processes entire run from ingest through artifacts; no inter-phase distribution.

---

## LLM Safety & Scope
- Only `Department` and `Account Name` are LLM-enabled.
- No LLM on numeric, date, ID, or PII-bearing columns.
- Strict JSON contracts; bounded retries; low-confidence outputs quarantined.
- Mapping cache ensures repeatability and cost control; write-through on successful LLM decisions.

---

## Non-Goals (PoC)
- No semantic imputation from `Description` to fill missing IDs or codes.
- No RAG/vector retrieval; any future retrieval must be version-pinned and audited.
- No multi-tenant auth; documented as future work.

---

## Acceptance Criteria (System Level)
- Identical inputs yield identical artifact hashes.
- Every changed cell appears in the diff with source and reason.
- All non-conforming rows are quarantined with categorized reasons.
- API surfaces status and artifacts; README reproduces the run end-to-end.

---
