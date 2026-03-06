---
title: "TensorDB v0.3.0: A Rust-Native Bitemporal Ledger Database with Full SQL and PostgreSQL Wire Protocol"
published: true
description: "TensorDB v0.3.0 ships 6 phases of features including recursive CTEs, Raft consensus, mTLS, column-level encryption, and 276ns point reads."
tags: [rust, database, sql, opensource]
---

# TensorDB v0.3.0: A Rust-Native Bitemporal Ledger Database

If you have ever needed to answer the question *"what did we know, and when did we know it?"* — across financial records, audit trails, medical histories, or regulatory filings — you have probably hacked together a solution with `created_at` and `updated_at` columns, soft-deletes, and a prayer. TensorDB is built to make that question a first-class citizen of your database engine.

**TensorDB v0.3.0** is out, and it is a significant release. Six phases of work landed: full SQL completeness, advanced types, enterprise security, distributed consensus, WASM/FFI edge deployment, and a learned cost model. This post walks through what TensorDB is, how it works, and what is new.

## What Is TensorDB?

TensorDB is an **embeddable, bitemporal, append-only ledger database** written entirely in Rust. It supports:

- **Bitemporal data model** — every record carries both *system time* (`commit_ts`) and *business time* (`valid_from` / `valid_to`)
- **MVCC with immutable storage** — facts are never overwritten; updates create new facts, deletes create tombstones
- **Full SQL engine** — hand-written recursive descent parser, cost-based planner, vectorized execution
- **LSM-tree storage** — WAL → Memtable → SSTables with LZ4/Zstd compression and L0–L6 compaction
- **PostgreSQL wire protocol** — connect with `psql`, `pgAdmin`, `asyncpg`, or any Postgres client
- **Embeddable** — link directly into your Rust binary, or use Python/Node.js bindings

## The Bitemporal Model in 60 Seconds

Most databases track *one* timeline. Bitemporal databases track *two*:

| Dimension | Column | Question answered |
|-----------|--------|-------------------|
| System time | `commit_ts` | When did the database record this fact? |
| Business time | `valid_from` / `valid_to` | When was this fact true in the real world? |

```sql
-- What did our inventory system show last Tuesday?
SELECT * FROM inventory AS OF SYSTEM TIME '2026-03-01 09:00:00';

-- What was the contractual price on Jan 1, even if we corrected it later?
SELECT * FROM pricing VALID AT DATE '2026-01-01';

-- Full SQL:2011 temporal range
SELECT * FROM orders
FOR SYSTEM_TIME FROM '2026-01-01' TO '2026-03-01';
```

No extra tables, no shadow schemas, no application-layer bookkeeping.

## Quick Start

### Rust (embedded)

```toml
[dependencies]
tensordb = "0.3"
```

```rust
use tensordb::Database;

fn main() -> tensordb::Result<()> {
    let db = Database::open("./mydb")?;

    db.sql("CREATE TABLE accounts (
        id INTEGER PRIMARY KEY, owner TEXT, balance REAL
    )")?;

    db.sql("INSERT INTO accounts VALUES (1, 'alice', 10000.00)")?;

    // Time-travel query
    let rows = db.sql(
        "SELECT * FROM accounts AS OF SYSTEM TIME '2026-03-01'"
    )?;
    println!("{:?}", rows);
    Ok(())
}
```

### Python

```bash
pip install tensordb
```

```python
from tensordb import PyDatabase

db = PyDatabase.open("./mydb")
db.sql("CREATE TABLE trades (id INT, symbol TEXT, price REAL)")
db.sql("INSERT INTO trades VALUES (1, 'AAPL', 182.50), (2, 'TSLA', 245.00)")
rows = db.sql("SELECT * FROM trades WHERE price > 200")
print(rows)
```

### Connect with psql

```bash
cargo run -p tensordb-server -- --data-dir ./mydb --port 5433
psql -h localhost -p 5433 -d mydb
```

## Performance

| Operation | TensorDB | SQLite (WAL mode) |
|-----------|----------|-------------------|
| Point read | **276 ns** | ~400 ns |
| Point write | **1.9 µs** | ~15 µs |
| Batch insert (10k rows) | ~18 ms | ~35 ms |

The fast write path uses atomic CAS for lock-free writes. Direct reads bypass shard actors via `ShardReadHandle` with `parking_lot::RwLock`.

## What Is New in v0.3.0

### SQL Completeness
OFFSET, IF EXISTS, multi-value INSERT, FULL OUTER JOIN, RETURNING on UPDATE/DELETE, subqueries (IN, EXISTS, scalar), upsert (ON CONFLICT), persistent sessions.

### Advanced SQL
Native date/time types, JSON operators (`->`, `->>`, `@>`), generated columns, recursive CTEs, foreign keys, materialized views, triggers, user-defined functions.

### Performance
Zstd compression policies, batch write optimization, external merge sort, expression compilation, query parallelism with rayon.

### Enterprise Security
Audit log tamper detection (SHA-256 hash chains), mTLS, encryption key rotation, column-level encryption (AES-256-GCM).

### Distributed
Raft consensus via gRPC, S3 storage backend, WAL replication, WASM/FFI edge deployment.

### Category Differentiation
Learned cost model, anomaly detection, graph queries, in-database ML (linear/logistic regression).

## Links

- **GitHub**: [tensor-db/TensorDB](https://github.com/tensor-db/TensorDB)
- **Docs**: [tensor-db.github.io/TensorDB](https://tensor-db.github.io/TensorDB/)
- **crates.io**: [tensordb](https://crates.io/crates/tensordb)
- **PyPI**: [tensordb](https://pypi.org/project/tensordb/)
- **npm**: [tensordb](https://www.npmjs.com/package/tensordb)

Contributions, benchmarks, and feedback are welcome. Open an issue or discussion on GitHub.
