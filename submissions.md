# Submission Drafts

## Hacker News (Show HN)

**Title:** Show HN: TensorDB – Bitemporal ledger database in Rust with full SQL and 276ns reads

**URL:** https://github.com/tensor-db/TensorDB

**Comment:**
Hi HN, I built TensorDB — an embeddable bitemporal ledger database written in Rust.

Every record carries two timestamps: when the database recorded it (system time) and when it was true in the real world (business time). This means you can time-travel to any past state with `SELECT * FROM orders AS OF SYSTEM TIME '2026-01-15'` without maintaining separate audit tables.

Key design decisions:
- Append-only immutable storage — facts are never overwritten, deletes create tombstones
- LSM-tree engine (WAL → Memtable → SSTables) with LZ4/Zstd compression
- Hand-written recursive descent SQL parser with cost-based planner
- PostgreSQL wire protocol — connect with psql or any Postgres client
- 276ns point reads via lock-free direct read handles
- Python (PyO3) and Node.js (napi-rs) bindings

v0.3.0 adds: recursive CTEs, foreign keys, materialized views, Raft consensus, mTLS, column-level encryption, and in-database ML.

Licensed under PolyForm Noncommercial. Happy to answer questions.

---

## Reddit r/rust

**Title:** TensorDB v0.3.0 – Bitemporal ledger database with MVCC, full SQL, 276ns reads, and PostgreSQL wire protocol

**Body:**
I've been building TensorDB, a bitemporal ledger database written entirely in Rust. v0.3.0 just shipped with 40+ new features across 6 phases.

**What makes it different:** Every record has two time dimensions — system time (when recorded) and business time (when true). You can query any historical state with SQL:2011 temporal clauses. Nothing is ever deleted — the storage is append-only with MVCC.

**Architecture:**
- LSM-tree storage engine (WAL → Memtable → SSTables)
- Hand-written recursive descent SQL parser
- Cost-based query planner with learned cost model
- Vectorized execution (1024-row batches)
- 4 shards by default, lock-free fast write path

**v0.3.0 highlights:**
- Full SQL: recursive CTEs, subqueries, FULL OUTER JOIN, upsert, triggers, UDFs
- Performance: Zstd compression, expression compilation, rayon parallelism
- Security: mTLS, column-level encryption, audit log hash chains
- Distributed: Raft consensus, S3 backend, WAL replication
- Bindings: Python (PyO3), Node.js (napi-rs), pgwire server

Links:
- GitHub: https://github.com/tensor-db/TensorDB
- Docs: https://tensor-db.github.io/TensorDB/
- crates.io: https://crates.io/crates/tensordb (pending publish)

Happy to discuss architecture decisions, benchmarks, or take feature requests.

---

## Reddit r/database

**Title:** TensorDB v0.3.0 – An embeddable bitemporal database with full SQL and PostgreSQL wire protocol

**Body:**
TensorDB is a bitemporal ledger database built in Rust. Every record carries both system time and business time, enabling time-travel queries and historical corrections without application-layer bookkeeping.

**Use cases:** financial ledgers, audit trails, regulatory compliance, healthcare records — anywhere you need to answer "what did we know, and when did we know it?"

**Key features:**
- Full SQL engine with SQL:2011 temporal clauses
- PostgreSQL wire protocol (connect with psql)
- 276ns point reads, 1.9µs point writes
- MVCC with append-only immutable storage
- LSM-tree engine with LZ4/Zstd compression
- Python and Node.js bindings

GitHub: https://github.com/tensor-db/TensorDB
Docs: https://tensor-db.github.io/TensorDB/
