<p align="center">
  <img src="assets/spectradb-logo.svg" alt="SpectraDB" width="560">
</p>

<p align="center">
  <strong>An append-only, bitemporal ledger database with MVCC snapshot reads and a SQL query interface.</strong>
</p>

<p align="center">
  <a href="https://github.com/spectra-db/SpectraDB/actions"><img src="https://github.com/spectra-db/SpectraDB/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/rust-stable-orange.svg" alt="Rust"></a>
</p>

---

SpectraDB is a single-node embedded database that treats every write as an immutable fact. It separates the **system timeline** (when data was recorded) from the **business-valid timeline** (when data was true), giving you built-in time travel and auditability with zero application-level bookkeeping.

## Key Features

- **Immutable Fact Ledger** — Write-ahead log with CRC-framed records. Data is never overwritten; updates create new versions.
- **MVCC Snapshot Reads** — Query any past state with `AS OF <commit_ts>`.
- **Bitemporal Filtering** — Separate system and valid-time dimensions with `VALID AT <valid_ts>`.
- **LSM Storage Engine** — Memtable, prefix-compressed SSTables with bloom filters, block index, and mmap reads. Multi-level compaction with configurable size budgets.
- **Block & Index Caching** — LRU caches for SSTable blocks and indexes with configurable memory budgets.
- **SQL Interface** — `CREATE TABLE` (legacy JSON or typed columns), `INSERT`, `SELECT`, `UPDATE`, `DELETE`, `JOIN` (inner/left/right/cross), `GROUP BY`, `HAVING`, subqueries, CTEs, window functions (`ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LEAD`, `LAG`), views, indexes, transactions, `EXPLAIN`, `COPY`.
- **Write Batch API** — Atomic multi-key writes with a single WAL frame for bulk ingest.
- **Embedded AI Insight Runtime** — Optional async copilot pipeline (`--ai-auto-insights`) that runs natively in-process, consumes change feeds, synthesizes operational insights, and stores them as immutable internal facts.
- **Streaming Change Feeds** — Subscribe to prefix-filtered change events for real-time downstream consumers.
- **Full-Text Search** — Inverted index facet with tokenizer, stemmer, and posting list intersection.
- **Time-Series** — Bucketed storage with range queries and downsampling (avg/min/max/sum).
- **Interactive Shell** — TAB completion, persistent history, table/line/JSON output modes.
- **Language Bindings** — Python (PyO3) and Node.js (napi-rs) bindings for cross-language access.
- **Optional Native Acceleration** — C++ kernels behind `--features native` via `cxx`, with pure Rust as the default.
- **Optional io_uring** — Linux async I/O for WAL writes and SSTable reads behind `--features io-uring`.
- **Optional SIMD** — Hardware-accelerated bloom probes and checksums behind `--features simd`.

## Who Is This For?

<table>
<tr>
<td width="50%" valign="top">

### Audit & Compliance Systems
Regulations demand provable data history. SpectraDB's append-only ledger with bitemporal queries lets you reconstruct the exact state of any record at any point in time — no application-level versioning required. Use `AS OF` and `VALID AT` to answer "what did we know, and when did we know it?"

</td>
<td width="50%" valign="top">

### Event-Sourced Applications
Use SpectraDB as the append-only event store behind CQRS or event-driven architectures. Write batches provide atomic multi-event commits. Streaming change feeds push events to downstream consumers in real time.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### Temporal Data Management
Model entities with business-valid time ranges — contracts, policies, price schedules — and query them with full SQL including joins, window functions, and aggregates. Ask "what was true on date X?" without building custom versioning logic.

</td>
<td width="50%" valign="top">

### Embedded & Local-First Apps
Ship a full temporal database as a Rust library, Python package, or Node.js module — no server process, no network, no Docker. Ideal for IoT gateways, on-device analytics, and desktop applications.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### Real-Time Monitoring & IoT
The time-series facet handles high-frequency metric ingestion with bucketed storage and downsampling. Combine with change feeds to trigger alerts and downstream processing as data arrives.

</td>
<td width="50%" valign="top">

### Search-Enabled Applications
The full-text search facet provides tokenized, stemmed inverted indexes over document fields. Build search experiences alongside temporal queries without bolting on a separate search service.

</td>
</tr>
</table>

**SpectraDB is a good fit if you need:**
- A complete, queryable history of every change — not just the latest state
- Time travel queries without maintaining versioning logic in your app
- An embedded database that deploys as a library, not a service
- Bitemporal semantics (system time vs. business-valid time) as a first-class primitive
- Real-time change feeds and full-text search in a single embedded engine

**SpectraDB is not the right choice if you need:**
- Distributed multi-node replication
- Petabyte-scale OLAP workloads (consider DuckDB or ClickHouse)
- A drop-in replacement for PostgreSQL or MySQL

## Quickstart

```bash
# Build
cargo build -p spectradb-cli

# Launch interactive shell
cargo run -p spectradb-cli -- --path ./mydb

# Launch shell with AI auto-insights enabled
cargo run -p spectradb-cli -- --path ./mydb --ai-auto-insights

# Use Tier-0 in-core AI runtime
cargo run -p spectradb-cli -- --path ./mydb --ai-auto-insights \
  --ai-batch-window-ms 20 \
  --ai-batch-max-events 16
```

```sql
-- Legacy schema (JSON documents)
CREATE TABLE events (pk TEXT PRIMARY KEY);

INSERT INTO events (pk, doc) VALUES ('evt-1', '{"type":"signup","user":"alice"}');
INSERT INTO events (pk, doc) VALUES ('evt-2', '{"type":"purchase","user":"bob","amount":49.99}');

-- Typed column schema
CREATE TABLE accounts (id INTEGER PRIMARY KEY, name TEXT NOT NULL, balance REAL);

INSERT INTO accounts (id, name, balance) VALUES (1, 'alice', 1000.0);
INSERT INTO accounts (id, name, balance) VALUES (2, 'bob', 500.0);

-- Query latest state
SELECT pk, doc FROM events ORDER BY pk LIMIT 10;

-- Time travel: read state as of commit 1
SELECT doc FROM events WHERE pk='evt-1' AS OF 1;

-- Bitemporal: what was valid at a specific point
SELECT doc FROM events VALID AT 1000;

-- Aggregates and grouping
SELECT count(*), sum(doc.amount) FROM events GROUP BY doc.type HAVING count(*) > 1;

-- Window functions
SELECT pk, doc, ROW_NUMBER() OVER (ORDER BY pk) AS rn FROM events;

-- Joins
SELECT pk, doc FROM events JOIN accounts ON events.pk=accounts.pk ORDER BY pk ASC;

-- UPDATE and DELETE
UPDATE events SET doc = '{"type":"refund","user":"bob"}' WHERE pk = 'evt-2';
DELETE FROM events WHERE pk = 'evt-1';

-- Transactions
BEGIN;
INSERT INTO events (pk, doc) VALUES ('evt-3', '{"type":"refund","user":"bob"}');
COMMIT;

-- Shell AI ops commands
.ai_status
.ai table/events/evt-3 5
.ai_timeline table/events/evt-3 20
.ai_correlate table/events/evt-3 20
# grab an insight_id from .ai output first
.ai_explain <insight-id>
```

Native AI note:
- AI auto-insights run fully in-process in the core engine.
- No external model server/process is required.

## Architecture

SpectraDB is organized around three core principles: **immutable truth** (the append-only ledger), **temporal indexing** (bitemporal metadata on every fact), and **faceted queries** (pluggable query planes over the same data).

```mermaid
graph TB
    subgraph Client Layer
        CLI[Interactive Shell]
        API[Rust API<br/><code>db.sql&#40;...&#41;</code>]
        PY[Python Bindings<br/>PyO3]
        NODE[Node.js Bindings<br/>napi-rs]
    end

    subgraph Query Engine
        Parser[SQL Parser<br/>expressions · CTEs · window funcs]
        Executor[Query Executor<br/>scans · joins · aggregates · windows]
        Planner[Explain Planner]
    end

    subgraph Facet Layer
        RF[Relational Facet<br/>tables · views · indexes · typed schema]
        FTS[Full-Text Search<br/>inverted index · stemmer]
        TS[Time-Series Facet<br/>buckets · downsample]
    end

    subgraph Shard Engine
        direction LR
        S0[Shard 0]
        S1[Shard 1]
        SN[Shard N]
    end

    CF[Change Feeds<br/>prefix-filtered subscriptions]

    subgraph Storage Engine
        WAL[Write-Ahead Log<br/>CRC-framed · batch writes · fsync]
        MT[Memtable<br/>sorted in-memory map]
        BC[Block & Index Cache<br/>LRU · configurable budgets]
        SST[SSTables<br/>bloom · prefix compression · mmap]
        C[Multi-Level Compaction<br/>L0 → L1 → ... · size-budgeted]
    end

    MF[Manifest<br/>per-level file tracking]

    CLI --> Parser
    API --> Parser
    PY --> API
    NODE --> API
    Parser --> Executor
    Executor --> RF
    Executor --> Planner
    RF --> S0
    RF --> S1
    RF --> SN
    FTS --> S0
    TS --> S0
    S0 --> CF
    S1 --> CF
    SN --> CF
    S0 --> WAL
    S1 --> WAL
    SN --> WAL
    WAL --> MT
    MT -->|flush| SST
    SST --> BC
    SST --> C
    S0 --> MF
    S1 --> MF
    SN --> MF
```

### Write Path

1. **Route** — Key is hashed to a shard (`hash(key) % shard_count`).
2. **Log** — Fact is appended to the shard's WAL with a CRC frame (single or batch).
3. **Notify** — Matching change feed subscribers receive the event.
4. **Buffer** — Entry is inserted into the in-memory memtable.
5. **Flush** — When the memtable exceeds `memtable_max_bytes`, it is frozen and written as a prefix-compressed SSTable.
6. **Compact** — Multi-level compaction promotes SSTables through L0 → L1 → ... with size-budgeted thresholds. All temporal versions are preserved.

### Read Path

1. **Cache Check** — LRU block and index caches serve hot data without disk I/O.
2. **Bloom Check** — If the bloom filter says the key is absent, skip the SSTable.
3. **Memtable Scan** — Check the active memtable for the latest version.
4. **Level Lookup** — L0: search all files newest-first. L1+: binary search for the single overlapping file per level.
5. **Temporal Filter** — Apply `AS OF` (system time) and `VALID AT` (business time) predicates.
6. **Merge** — Return the most recent version satisfying all filters.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Append-only writes | Immutability simplifies recovery, enables time travel, eliminates in-place update corruption |
| Single writer per shard | Avoids fine-grained locking while allowing parallel writes across shards |
| Bitemporal timestamps | Separates "when recorded" from "when true" — a requirement for audit and compliance workloads |
| Multi-level compaction | Size-budgeted leveling reduces read amplification while preserving all temporal versions |
| Block/index caching | LRU caches reduce disk I/O for hot working sets without requiring mmap for everything |
| Dual schema modes | Legacy JSON documents for flexibility; typed columns for structure and performance |

## Performance

SpectraDB ships with a built-in benchmark harness:

```bash
cargo run -p spectradb-cli -- --path /tmp/bench bench \
  --write-ops 100000 --read-ops 50000 --keyspace 20000 --read-miss-ratio 0.20
```

Sample numbers (single machine, sanity run):

| Metric | Value |
|--------|-------|
| Write throughput | ~4,500 ops/s |
| Read p50 latency | ~530 µs |
| Read p95 latency | ~890 µs |
| Read p99 latency | ~1,030 µs |

Tuning knobs: `--wal-fsync-every-n-records`, `--memtable-max-bytes`, `--sstable-block-bytes`, `--bloom-bits-per-key`, `--shard-count`. See [perf.md](perf.md) for details.

AI overhead gate (CI/local):

```bash
./scripts/ai_overhead_gate.sh
```

Release AI overhead report:

```bash
./scripts/release_ai_report.sh
```

Executable docs note:
- Quickstart SQL/AI examples are validated in [tests/readme_examples.rs](tests/readme_examples.rs).

## Completed Releases

<details>
<summary><strong>v0.2 — Query Engine</strong> ✅</summary>

- [x] Expression AST with full precedence parsing (OR, AND, NOT, comparisons, arithmetic)
- [x] General `WHERE` clauses with comparison operators and field access
- [x] `UPDATE` and `DELETE` with temporal-aware semantics
- [x] General-purpose `JOIN` (inner, left, right, cross) with arbitrary ON clauses
- [x] Richer aggregates: `SUM`, `AVG`, `MIN`, `MAX`
- [x] `GROUP BY` on arbitrary expressions
- [x] `HAVING` clause
</details>

<details>
<summary><strong>v0.3 — Storage & Performance</strong> ✅</summary>

- [x] Multi-level compaction with size-budgeted leveling (L0 → L1 → ... → L7)
- [x] Block and index caching with configurable memory budgets (LRU)
- [x] Prefix compression and restart points in SSTable blocks (V2 format)
- [x] Write-batch API for atomic multi-key bulk ingest
- [x] SIMD-accelerated bloom probes and checksums (behind `--features simd`)
</details>

<details>
<summary><strong>v0.4 — SQL Surface & Developer Experience</strong> ✅</summary>

- [x] Subqueries, CTEs (`WITH ... AS`)
- [x] Window functions: `ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LEAD`, `LAG`
- [x] Typed column schema with DDL enforcement (`INTEGER`, `REAL`, `TEXT`, `BOOLEAN`, `BLOB`)
- [x] Columnar row encoding with null bitmaps
- [x] Index-backed query execution
- [x] `COPY` for bulk import/export
- [x] Python bindings (PyO3)
- [x] Node.js bindings (napi-rs)
</details>

<details>
<summary><strong>v0.5 — Ecosystem</strong> ✅</summary>

- [x] Full-text search facet (inverted index, tokenizer, stemmer)
- [x] Time-series facet (bucketed storage, downsampling, range queries)
- [x] Streaming change feeds with prefix-filtered subscriptions
- [x] io_uring async I/O for WAL writes and SSTable reads (behind `--features io-uring`)
- [x] Comparative benchmark harness
</details>

## Roadmap

### v0.6 — Correctness & Storage Hardening
**Fix critical storage bugs and wire unused subsystems into the engine.**
- [ ] Persist per-level file metadata in manifest (fix L0-reload-on-restart bug)
- [ ] Wire block/index caches into SSTable reader (currently allocated but unused)
- [ ] Implement immutable memtable pipeline (freeze → queue → flush, currently dead code)
- [ ] Fix numeric ORDER BY (currently lexicographic: "10" < "9")
- [ ] Fix window function evaluation order (compute windows before LIMIT, not after)
- [ ] Transaction-local reads: `SELECT` inside `BEGIN..COMMIT` sees own staged writes
- [ ] Typed DELETE tombstone record with compaction GC (replace empty-value tombstone)
- [ ] Wire columnar row encoding into typed table write/read path (currently stores JSON)
- [ ] Wire io_uring backend into WAL and SSTable read paths (currently compiled but disconnected)
- [ ] SSTable binary search correctness: fix `get_visible` index probe for internal key format
- [ ] WAL rotation: file-per-generation with automatic archival instead of single-file truncation
- [ ] Crash recovery fuzzing: random kill during flush/compaction with data integrity verification

### v0.7 — Core AI Intelligence
**Evolve the Tier-0 AI runtime from keyword matching to a real reasoning engine.**
- [ ] Multi-pass enrichment graph: tokenize → classify → score → correlate → synthesize
- [ ] Pluggable model backend trait (`ModelBackend`) with ONNX Runtime and HTTP options
- [ ] Anomaly detection: statistical baseline per key prefix with z-score deviation alerts
- [ ] Pattern learning: track recurring tag/risk sequences per cluster to predict incidents
- [ ] Cross-shard correlation: join events across shards by time window and tag overlap
- [ ] AI-driven compaction hints: prioritize compaction of hot key ranges based on access patterns
- [ ] Adaptive batch tuning: auto-adjust `batch_window_ms` and `batch_max_events` based on event rate
- [ ] AI query advisor: `EXPLAIN AI` suggests indexes, schema changes, query rewrites
- [ ] Configurable risk thresholds per key prefix (e.g., `payments/` gets lower risk tolerance)
- [ ] AI insight retention policy: TTL-based expiration of old `__ai/` facts

### v0.8 — Query Optimizer & Execution Engine
**Replace full-table scans with a real query planner.**
- [ ] Cost-based query optimizer with table statistics (row count, value distribution histograms)
- [ ] Predicate pushdown: push WHERE clauses to SSTable block scan level
- [ ] Secondary index implementation: B-tree index on arbitrary columns with automatic maintenance
- [ ] Composite index support: multi-column indexes with prefix matching
- [ ] Index-only scans: return results directly from index without hitting data blocks
- [ ] Parallel query execution: fan out scan/aggregate across shards, merge results
- [ ] Hash join and sort-merge join implementations (replace nested-loop for large tables)
- [ ] Prepared statements with parameterized queries and plan caching
- [ ] `EXPLAIN ANALYZE` with runtime statistics (rows scanned, blocks read, cache hits)
- [ ] External merge sort for ORDER BY on datasets larger than memory
- [ ] Predicate compilation: compile hot WHERE clauses to native code via cranelift

### v0.9 — Full SQL & Analytics Surface
**Close the gap with PostgreSQL/DuckDB for analytics workloads.**
- [ ] Materialized views with incremental refresh on change feed
- [ ] Common Table Expressions with recursive support (`WITH RECURSIVE`)
- [ ] `UNION`, `INTERSECT`, `EXCEPT` set operations
- [ ] Correlated subqueries in WHERE and SELECT
- [ ] `CASE WHEN ... THEN ... ELSE ... END` expression
- [ ] `CAST(expr AS type)` with type coercion rules
- [ ] Additional scalar functions: `SUBSTR`, `TRIM`, `REPLACE`, `CONCAT`, `ROUND`, `CEIL`, `FLOOR`, `COALESCE`, `NULLIF`, `NVL`
- [ ] Date/time functions: `NOW()`, `DATE()`, `EXTRACT()`, `DATE_TRUNC()`, `INTERVAL` arithmetic
- [ ] Additional aggregate functions: `STDDEV`, `VARIANCE`, `FIRST`, `LAST`, `COUNT(DISTINCT ...)`
- [ ] Approximate aggregates: HyperLogLog for `APPROX_COUNT_DISTINCT`, t-digest for percentiles
- [ ] Additional window functions: `SUM() OVER`, `AVG() OVER`, `NTILE`, `FIRST_VALUE`, `LAST_VALUE`
- [ ] Window function frames: `ROWS BETWEEN N PRECEDING AND M FOLLOWING`
- [ ] JSON/JSONB native column type with `->`, `->>` operators and GIN path indexing
- [ ] Array column type with `ANY`, `ALL`, `UNNEST` operators
- [ ] `INSERT ... ON CONFLICT DO UPDATE` (upsert)
- [ ] `INSERT ... RETURNING`, `UPDATE ... RETURNING`, `DELETE ... RETURNING`
- [ ] `CREATE TABLE ... AS SELECT` (CTAS)
- [ ] User-defined functions via Rust plugin API (`CREATE FUNCTION`)
- [ ] Quoted identifiers and case-sensitive column names

### v0.10 — FTS & Time-Series SQL Integration
**Wire existing facets into the SQL surface for unified querying.**
- [ ] `CREATE FTS INDEX <name> ON <table> (<columns>)` DDL
- [ ] `MATCH(column, 'query')` function in WHERE clauses with relevance scoring
- [ ] FTS posting list maintenance on INSERT/UPDATE/DELETE
- [ ] BM25 relevance ranking with configurable k1/b parameters
- [ ] Phrase search, proximity search, boolean operators (`+`, `-`, `"..."`)
- [ ] `CREATE TIMESERIES TABLE <name> (ts TIMESTAMP, value REAL, tags MAP)` DDL
- [ ] `SELECT bucket(ts, '1h'), avg(value) FROM metrics WHERE ts BETWEEN ... GROUP BY bucket` syntax
- [ ] Time-series downsampling policies: automatic rollup from raw → 1m → 1h → 1d
- [ ] Retention policies: `ALTER TABLE SET RETENTION 90d` with automatic purge
- [ ] `SELECT ... FROM events MATCH 'search term' WHERE ts > NOW() - INTERVAL '1h'` — unified temporal + search

### v0.11 — Streaming, Change Data Capture & Event Processing
**Make SpectraDB the source of truth for event-driven architectures.**
- [ ] Durable change feed cursors: resume from last-acknowledged position after restart
- [ ] Exactly-once delivery semantics with consumer group coordination
- [ ] Change feed projections: subscribe to specific columns, not full documents
- [ ] Change feed transformations: filter/map/aggregate in-database before delivery
- [ ] Webhook delivery: HTTP POST to external endpoints on change feed match
- [ ] Outbox pattern: `CREATE OUTBOX ON <table> DELIVER TO <endpoint>` DDL
- [ ] Kafka Connect sink connector for streaming to Kafka/Redpanda
- [ ] gRPC streaming API for high-throughput change feed consumption
- [ ] Materialized aggregates: continuously updated counters/gauges driven by change feeds
- [ ] Temporal event joins: correlate events across tables within time windows

### v0.12 — Enterprise Security & Multi-Tenancy
**Production-grade security for regulated enterprise environments.**
- [ ] Encryption at rest: AES-256-GCM per SSTable block with key rotation
- [ ] Encryption in transit: TLS 1.3 for all network APIs (gRPC, wire protocol)
- [ ] Role-based access control: `CREATE ROLE`, `GRANT`, `REVOKE` on tables/views
- [ ] Row-level security: `CREATE POLICY ON <table> USING (tenant_id = current_tenant())`
- [ ] Column-level encryption: `CREATE TABLE ... (ssn TEXT ENCRYPTED)` with per-column keys
- [ ] Audit log: immutable record of all DDL and DML operations with user attribution
- [ ] Multi-tenant isolation: logical database separation with resource quotas per tenant
- [ ] Connection pooling with per-tenant connection limits
- [ ] LDAP/OIDC authentication integration
- [ ] SOC 2 / HIPAA compliance mode: enforce encryption, audit logging, key management

### v0.13 — Distributed & High Availability
**Scale beyond single-node for enterprise deployments.**
- [ ] WAL replication: synchronous leader-follower with configurable quorum
- [ ] Read replicas: route read-only queries to followers with bounded staleness
- [ ] Snapshot-based backup: `BACKUP TO 's3://...'` / `RESTORE FROM '...'`
- [ ] Point-in-time recovery: replay WAL from any backup to any timestamp
- [ ] Automatic leader election via Raft consensus
- [ ] Online shard rebalancing: split/merge shards without downtime
- [ ] Cross-datacenter replication with conflict resolution policies
- [ ] Geo-partitioning: route keys to specific regions based on prefix rules
- [ ] Connection routing: smart proxy that routes queries to appropriate replicas
- [ ] Cluster-wide schema DDL coordination

### v0.14 — Observability & Operations
**Enterprise-grade monitoring, debugging, and capacity planning.**
- [ ] Prometheus metrics endpoint: write/read throughput, latency histograms, cache hit rates
- [ ] OpenTelemetry tracing: per-query span with storage/compute breakdown
- [ ] Slow query log with configurable threshold and EXPLAIN output
- [ ] Live query dashboard: `SHOW PROCESSLIST`, `KILL <query_id>`
- [ ] Storage analytics: `SHOW TABLE STATS`, SSTable count/size per level, compaction history
- [ ] Memory usage breakdown: memtable, block cache, index cache, WAL buffers
- [ ] Automatic vacuum: reclaim space from tombstoned temporal versions beyond retention
- [ ] Online DDL: `ALTER TABLE` without locking writes
- [ ] Hot config reload: change compaction thresholds, cache sizes without restart
- [ ] Health check endpoint with readiness/liveness probes for Kubernetes

### v0.15 — Language Ecosystem & Wire Protocol
**First-class support for every major language and tool.**
- [ ] PostgreSQL wire protocol (pgwire): connect with `psql`, any Postgres driver, BI tools
- [ ] Python SDK: full API including `scan_prefix`, `write_batch`, `subscribe`, AI methods, async support
- [ ] Node.js SDK: full API with native JS objects (not JSON strings), TypeScript types, async iterators
- [ ] Go SDK via CGo or gRPC client
- [ ] Java/JVM SDK via JNI or gRPC client
- [ ] C/C++ SDK via FFI with stable ABI
- [ ] REST API: HTTP/JSON interface for language-agnostic access
- [ ] GraphQL API: schema auto-generated from table DDL
- [ ] JDBC/ODBC drivers for BI tool integration (Tableau, Power BI, Metabase)
- [ ] SQLAlchemy dialect for Python ORM integration
- [ ] Prisma adapter for Node.js ORM integration
- [ ] `spectradb-wasm`: compile core to WASM for browser and edge runtimes

### v0.16 — Advanced AI & ML Integration
**Make SpectraDB the AI-native database for enterprise ML pipelines.**
- [ ] Vector column type with HNSW index for similarity search
- [ ] `SELECT * FROM docs ORDER BY embedding <-> query_vector LIMIT 10` — vector search syntax
- [ ] Hybrid search: combine FTS relevance + vector similarity + temporal recency
- [ ] Embedding generation: auto-embed text columns via pluggable model (ONNX, OpenAI, local)
- [ ] Feature store: versioned feature tables with point-in-time-correct joins for ML training
- [ ] Model registry: store and version ML models as BLOB columns with metadata
- [ ] Inference UDF: `SELECT predict(model_name, features) FROM table` — in-database inference
- [ ] AI-powered schema migration: suggest schema changes based on query patterns and data evolution
- [ ] RAG pipeline primitive: `SELECT rag_answer('question', corpus_table, embedding_col)` end-to-end
- [ ] Training data export: `COPY TO 'dataset.parquet' FORMAT PARQUET` with temporal filtering for reproducible training sets
- [ ] Drift detection: continuous monitoring of data distribution changes in feature columns

### v0.17 — Performance & Scale
**Push single-node performance to hardware limits.**
- [ ] Vectorized query execution: process columns in batches of 1024 with SIMD
- [ ] Column-oriented storage format: optional columnar SSTables for analytics workloads
- [ ] Parallel compaction: concurrent compaction across levels with I/O scheduling
- [ ] Direct I/O (`O_DIRECT`) bypass for WAL and SSTable writes on Linux
- [ ] Memory-mapped writes with `MAP_POPULATE` for SSTable builds
- [ ] Zero-copy reads: return references into mmap'd SSTables without allocation
- [ ] Lock-free memtable: replace `BTreeMap` with a concurrent skip list
- [ ] Partition pruning: skip shards that provably don't contain matching keys
- [ ] Bloom filter hierarchy: per-level bloom filters for range queries
- [ ] Adaptive compression: LZ4 for L0-L2 (speed), Zstd for L3+ (ratio)
- [ ] Huge page support for large caches and memtables
- [ ] NUMA-aware shard placement: pin shards to specific CPU/memory nodes

### v1.0 — Stable Release
**Production-ready with stability guarantees.**
- [ ] Stable on-disk format with forward-compatibility guarantees (format version 1.0)
- [ ] Backward-compatible WAL format with version negotiation
- [ ] Semantic versioning with breaking change policy
- [ ] Published crates on crates.io with API stability promise
- [ ] Published packages on PyPI and npm
- [ ] Comprehensive documentation site with tutorials, API reference, and migration guides
- [ ] Jepsen-style distributed correctness testing
- [ ] Chaos engineering suite: random kill, disk corruption, network partition
- [ ] Long-term support (LTS) policy: 2-year security patches for each major release
- [ ] Conformance test suite for wire protocol compatibility

See [design.md](design.md) for the full architecture specification.

## Project Structure

```
spectradb/
├── crates/
│   ├── spectradb-core/        # Storage engine, SQL parser/executor, facets
│   │   └── src/
│   │       ├── ai/            # Tier-0 in-core AI runtime
│   │       ├── engine/        # Database, shard runtime, change feeds
│   │       ├── storage/       # SSTable, WAL, compaction, levels, cache, columnar
│   │       ├── sql/           # Parser, executor, evaluator, planner
│   │       ├── facet/         # Relational, full-text search, time-series
│   │       ├── io/            # io_uring async I/O (optional)
│   │       └── util/          # Bloom filters, varint, checksums
│   ├── spectradb-cli/         # Interactive shell and CLI commands
│   ├── spectradb-native/      # Optional C++ acceleration (cxx)
│   ├── spectradb-python/      # Python bindings (PyO3 / maturin)
│   └── spectradb-node/        # Node.js bindings (napi-rs)
├── tests/                     # Integration tests (100+ tests)
├── benches/                   # Criterion benchmarks (basic + comparative)
├── scripts/                   # Benchmark matrix, overnight burn-in
├── design.md                  # Architecture deep dive
├── perf.md                    # Performance notes and tuning guide
└── TEST_PLAN.md               # Validation strategy
```

## Documentation

| Document | Description |
|----------|-------------|
| [design.md](design.md) | Internal architecture, data model, storage format |
| [perf.md](perf.md) | Tuning knobs, benchmark methodology, optimization roadmap |
| [TEST_PLAN.md](TEST_PLAN.md) | Correctness, recovery, temporal, and soak test strategy |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup and contribution guidelines |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

## Building

```bash
# Pure Rust (default)
cargo test

# With C++ acceleration
cargo test --features native

# With SIMD-accelerated bloom probes and checksums
cargo test --features simd

# With io_uring async I/O (Linux only)
cargo test --features io-uring

# Run benchmarks
cargo bench

# Build Python bindings
cd crates/spectradb-python && maturin develop

# Build Node.js bindings
cd crates/spectradb-node && npm run build
```

## Contributing

We welcome contributions. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License

SpectraDB is licensed under the [MIT License](LICENSE).
