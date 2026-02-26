# SpectraDB Design

## 1) Purpose and Model

SpectraDB separates immutable truth from query projections:
- Core truth: append-only bitemporal fact ledger.
- Facets: query planes built over the same truth.

Current builds ship relational, full-text, and time-series facets while preserving a layout that supports future vector/graph facets.

## 2) Data Model

Each fact version has:
- `user_key`
- `commit_ts` (system/transaction timeline)
- `valid_from`, `valid_to` (business-valid timeline)
- payload (`doc` bytes)

Internal key format:

```text
internal_key = user_key || 0x00 || commit_ts(be_u64) || kind(u8)
```

MVP `kind`:
- `0`: PUT

## 3) Semantics

### MVCC
- Each shard owns a monotonic `u64` commit counter.
- `PUT` assigns commit timestamp in shard writer loop.
- Read `AS OF T` sees newest committed version where `commit_ts <= T`.

### Bitemporal
- Valid interval is half-open: `[valid_from, valid_to)`.
- `VALID AT V` returns versions satisfying:

```text
valid_from <= V < valid_to
```

### Combined
- `AS OF` filters by commit timeline.
- `VALID AT` filters by domain-valid timeline.
- Final record is max visible `commit_ts` after both filters.

## 4) Storage Pipeline

```text
WAL append -> memtable -> SSTable flush (L0) -> multi-level compaction
```

### WAL format

Frame:

```text
MAGIC(u32) | len(u32) | crc32(u32) | payload[len]
```

Payload (`FactWrite`):
- internal_key length varint + bytes
- fact length varint + bytes
- metadata presence flags + optional fields

Recovery behavior:
- CRC mismatch: deterministic stop at last valid frame
- torn tail: deterministic stop at last valid frame

### Memtable
- Ordered map keyed by internal key.
- Write-optimized in-memory staging.
- Flushed when `memtable_max_bytes` threshold is exceeded.

### SSTable format

```text
Header:
  MAGIC(u32), VERSION(u32), block_size(u32)

Data blocks:
  [block_len u32][entries...]
  entry = [klen varint][vlen varint][key][value]

Index block (at end):
  count(u32), repeated(last_key, offset, len)

Bloom block (at end):
  bloom metadata + bitset bytes

Footer:
  index_offset(u64), bloom_offset(u64), footer_magic(u32)
```

Lookup path:
1. Bloom negative => fast not found
2. Binary-search index for candidate block start
3. Decode/scan candidate block entries and apply temporal visibility filters

### Manifest
- Tracks per-shard WAL/SSTables and next file id.
- JSON metadata for MVP readability and deterministic replay.
- Atomic update pattern:
  - write temp
  - fsync temp
  - rename to MANIFEST
  - fsync parent dir

## 5) Shard Execution Model

Shard id:

```text
shard = hash(user_key) % shard_count
```

Each shard has a dedicated actor loop with:
- WAL handle
- active memtable
- immutable memtables (flush queue)
- L0 SSTable list
- local commit counter

Why this model:
- single writer per shard avoids fine-grained lock contention,
- sequential WAL I/O per shard,
- straightforward correctness boundaries.

## 6) SQL / Relational Facet

### 6.1 Implemented SQL Surface

- DDL: `CREATE TABLE` (legacy + typed), `CREATE VIEW`, `CREATE INDEX`, `ALTER TABLE ADD COLUMN`, `DROP TABLE/VIEW/INDEX`, `SHOW TABLES`, `DESCRIBE`.
- DML: `INSERT`, `SELECT`, `UPDATE`, `DELETE`, transactional batches (`BEGIN/COMMIT/ROLLBACK`).
- Query constructs: `WHERE` expressions, `ORDER BY`, `LIMIT`, subqueries, CTEs, joins (`inner/left/right/cross`), `GROUP BY`, `HAVING`, window functions (`ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LEAD`, `LAG`).
- Temporal filters: `AS OF`, `VALID AT`.
- Bulk data path: `COPY` import/export.
- Plan visibility: `EXPLAIN`.

### 6.2 Execution Semantics and Boundaries

- `db.sql(...)` executes a semicolon-aware batch and returns the final statement result.
- Transaction staging remains call-scoped: open transaction state does not persist across separate `db.sql(...)` calls.
- Temporal correctness remains first-class for read and write paths (`AS OF`, `VALID AT`, version-preserving compaction).
- Indexes and typed schema metadata are persisted; planner/executor coverage is expanding but not yet a full cost-based implementation.

### 6.3 Storage Mapping

- table metadata key: `__meta/table/<table>`
- view metadata key: `__meta/view/<view>`
- index metadata key: `__meta/index/<table>/<index>`
- row key: `table/<table>/<pk>`
- row payload: typed columnar encoding or legacy JSON payload, depending on table mode

### 6.4 Gap Map (Current)

- Optimizer: no full cost-based optimizer yet.
- Parallel execution: no fully parallel SQL executor yet.
- Advanced analytics: partial catalog compared to DuckDB/Postgres ecosystems.
- Prepared-plan lifecycle: no persistent plan cache yet.

### 6.5 Near-Term Query Targets

- Cost-based planning with collected statistics.
- Parallel shard-aware scan/join/aggregate execution.
- Expanded analytical functions and operator-level runtime instrumentation.

## 7) Tier-0 AI Runtime (In-Core)

The AI runtime is embedded directly into the core engine and runs fully in-process.

- Enablement: `ai_auto_insights` config flag.
- Input: shard change-feed events.
- Batching: micro-batched by `ai_batch_window_ms` and `ai_batch_max_events`.
- Synthesis: native core inference/enrichment path (no external model server or subprocess).
- Persistence: insights are written back as immutable internal facts under `__ai/insight/...`.
- Isolation: AI internal writes are hidden from user change-feed subscribers.

Operational surfaces:
- `Database::ai_stats()`
- `Database::ai_insights_for_key(...)`
- CLI shell: `.ai_status`, `.ai <key> [limit]`

## 8) Native Integration (Optional)

Feature flag: `native`

Design:
- core remains pure Rust and fully functional,
- `spectradb-native` crate hosts C++ code and `cxx` bridge,
- trait boundary in `native_bridge.rs`:
  - `Hasher`
  - `Compressor`
  - `BloomProbe`

Current module:
- native hasher with call-counter instrumentation,
- deterministic equality tests vs Rust reference implementation.

## 9) Invariants Checklist

- No ACK before WAL append.
- Shard commit timestamps are monotonic.
- Visible read result is highest commit meeting temporal predicates.
- SSTables are immutable once published.
- Manifest state is atomically replaced.
- Recovery from manifest + WAL replay reproduces visible state.
- AI internal keys (`__ai/...`) never leak into user-facing change feeds.

## 10) Facet Subscription Direction

Facets consume fact streams from ledger append points:
- append event contains key, commit_ts, valid interval, payload,
- facet-specific indexes can be updated asynchronously,
- core ledger remains source of truth for reconciliation and rebuilds.
