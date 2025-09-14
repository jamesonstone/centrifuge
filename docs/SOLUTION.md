# Centrifuge — Technical Decisions & Rationale

Version: v1
Scope: PoC that reads recurring CSVs, normalizes them with rules-first logic, and applies contract-safe LLM assists only where rules are brittle. Optimized for determinism, auditability, and credibility with an AI-skeptical stakeholder.

## Problem Framing

- **Core need:** Consistent, reproducible cleaning of a monthly CSV export with minimal surprises.
- **Primary user:** Tech-savvy controller; **secondary stakeholder:** skeptical boss who demands transparency, reversibility, and stable outputs.
- **Design north star:** Rules do the heavy lifting. LLM augments only on messy text categories with strict safeguards.

## Architecture at a Glance

- **Multi-process monolith:** One repo, shared code. Two roles:
  - **API (`centrifuge-app`)**: thin HTTP surface. Enqueues work. Serves status and artifacts. Never blocks on long jobs.
  - **Worker pool (`centrifuge-worker-*`)**: each worker claims exactly one run and executes end-to-end.
- **State & artifacts:**
  - **Postgres**: source of truth for runs, settings, canonical mapping cache, and artifact indices.
  - **MinIO**: artifact store for cleaned outputs, diffs, audit trail, metrics, and manifest.
- **LLM access via LiteLLM proxy:** centralized routing and budget control. Temperature=0 and fixed seed for determinism.

## Data Flow (Single-Run Ownership)

1. Client submits CSV (upload or S3 URL).
2. API writes a `runs` record with pinned versions and options.
3. Worker atomically claims the run, heartbeats, executes pipeline, uploads artifacts, and finalizes state.
4. API returns status, metrics, and artifact links.

## Pipeline Phases

- **Ingest:** stream CSV, sniff delimiter/encoding, normalize headers, compute content hash.
- **Validation-1:** required columns, types, enums, regex, domain rule `debit_xor_credit`, business-key assessment.
- **Rules engine:** whitespace, casing, numeric cleaning, dates → ISO, enum maps, reference formatting, Transaction ID normalization, debit/credit sign fixes, text de-noise.
- **Residual planner:** identify unresolved issues **only** on LLM-enabled columns; batch by unique values; check cache first.
- **LLM step (contracted):** generate patch suggestions with confidence; apply only with preconditions.
- **Validation-2 & quarantine:** re-validate; isolate non-compliant rows with explicit categories.
- **Artifacts & summary:** persist outputs, metrics, and a human-readable run summary.

## Determinism & Reproducibility

- **Temperature=0**, **seed=42**, **prompt and model versions pinned in manifest**.
- **Idempotency key:** hash of input bytes + version bundle.
- **Preconditioned patches:** apply only if the cell still equals the expected `before` value.
- **Content-addressed storage:** artifact paths derived from content hash to guarantee identical runs produce identical URIs.

## Rules-First Philosophy

- Numeric, date, identifier, and code columns are handled **purely by rules**.
- LLM is **opt-in per column** and restricted to messy, categorical text where rules are brittle.
- **Current LLM-enabled columns:** Department, Account Name. Everything else is rule-only or blocked from prompts.

## Canonicalization Decisions

- **Dates:** ISO `YYYY-MM-DD`.
- **Text categories:** Title Case; curated enum lists for Departments and common Account Names.
- **Transaction ID:** canonical `TXN-<digits>`, minimum 3-digit padding, preserve existing leading zeros for longer values, no inference; duplicates/missing are quarantined.
- **Debit/Credit:** strip formatting; enforce non-negative amounts; repair unambiguous sign errors; otherwise quarantine.

## Business Keys vs Surrogate Keys

- **Surrogate key:** `row_uuid` (UUIDv7) added to every row; used in diffs, audits, and joins.
- **Business key:** `Transaction ID` validated for presence and uniqueness **if present on the majority**; violations quarantined.
- Rationale: stable joins without mutating domain identity; transparent handling of bad inputs.

## Quarantine Policy

- **No silent fixes:** if the system cannot meet constraints deterministically, the row goes to quarantine with a primary category:
  - `validation_failure`, `llm_contract_failure`, `low_confidence`, `edit_cap_exceeded`, `parse_error`.
- Quarantined rows carry full context (including Description) for human review.

## LLM Safety & Scope Control

- **Prompt exposure control:** columns marked `allow_in_prompt=false` are never sent to the model (IDs, amounts, dates, PII-adjacent fields).
- **Confidence floor & edit cap:** defaults 0.80 and 20% per LLM column. Low-confidence or over-cap edits are quarantined.
- **Mapping cache:** persistent Postgres table stores approved `variant → canonical` mappings with `model_id`, `prompt_version`, and source. Cache hits avoid LLM calls and ensure repeatability.

## Auditability

- **Per-cell audit events** for every change: before/after, rule or contract id, reason, confidence, timestamp, row_uuid.
- **Diff file** for human inspection.
- **Summary report** highlights % fixed by rules vs LLM, quarantine breakdown, cache hit rates, per-phase timings, and token/cost estimates when available.

## API Surface (PoC)

- Create run, check status, list/download artifacts.
- No auth in the PoC; future work notes JWT/API keys and tenant quotas.
- Run records include versions, options, and immutable links to artifacts.

## Worker Model & Scale

- **Worker pool**: any number of workers can run; each owns a full run end-to-end.
- **Atomic claim:** `FOR UPDATE SKIP LOCKED`.
- **Heartbeat & visibility timeout:** safe reclaim if a worker dies; idempotency avoids double edits.
- **Future path:** switch to Kafka/Redpanda for event-driven scaling while keeping Postgres as the source of truth.

## Observability & Health (PoC level)

- **Structured logs** include `run_id` and `worker_id`.
- **Health endpoint** proves process liveness; minimal DB ping for readiness.
- **Run metrics** captured in an artifact; external TSDB noted as future work.

## Security & Privacy Posture (PoC)

- **Least exposure:** only LLM-enabled columns are sent to prompts; sensitive columns are blocked.
- **No free-text imputation:** Description retained for context, not used to infer IDs or codes.
- **Secrets:** environment variables; never committed.

## Test Strategy

- **Golden tests:** sample dataset with expected cleaned and diff outputs.
- **Determinism tests:** identical runs produce identical hashes.
- **Adversarial tests:** malformed model output or injection attempts → quarantined.
- **Property tests:** invariants like `debit_xor_credit`, PK uniqueness, enum closure, ISO dates.

## Infrastructure Choices

- **Docker Compose** for a reproducible demo environment.
- **Postgres** with simple bootstrap SQL for PoC; migrations discussed for production.
- **MinIO** for S3-compatible artifact storage.
- **LiteLLM proxy** to centralize model routing and budget control; fail fast if credentials are missing in prod mode; deterministic dev stub optional.

## Why Not RAG (Now)

- **Scope:** the knowledge surface is small and structured (schemas, enum lists, mapping cache).
- **Determinism:** retrieval variance reduces reproducibility and trust.
- **Overhead:** vector infra and evaluation add complexity without material benefit to this PoC.
- **Future path:** pgvector-backed retrieval feeding curated few-shots, version-pinned in the manifest, still under strict contracts.

## Future Enhancements (Intentionally Deferred)

- **Event bus:** Redpanda/Kafka for `run.created` and `run.status.changed`, DLQ, and replay.
- **Human-in-the-loop:** approval queue for quarantines; promote approved mappings to the canonical cache.
- **Governance:** tenant quotas, spend budgets, model routing policies, and RBAC.
- **Schema inference:** opt-in, with explicit labeling in the manifest; never default.
- **Extended canonical lists:** admin UI to curate Departments/Accounts; audit trail for list changes.
- **Semantic assistance:** guarded use of Description via retrieval for anomaly hints, never for silent imputation.

## Trade-offs & Justifications

- **Rules before models:** maximizes predictability, minimizes cost.
- **Narrow LLM use:** improves messy text while retaining traceability and edit limits.
- **Surrogate key design:** stability for audit joins without altering business identity.
- **DB-backed queue in PoC:** simpler than Kafka, yet safely concurrent; clear migration path exists.
- **Content-addressed artifacts:** enables cache-like behavior for results and easy reproducibility proofs.

## What This Demonstrates

- **Product judgment:** explicit alignment with a skeptical stakeholder’s needs.
- **Engineering rigor:** deterministic design, auditability, and controlled LLM usage.
- **Operational thinking:** clear scaling, recovery, and future-proofing paths without overbuilding.
- **Data governance:** principled handling of IDs, PII exposure limits, and transparent quarantines.
