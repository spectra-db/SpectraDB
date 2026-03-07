---
title: "I Built a Database That Never Forgets — Here's Why"
published: true
description: "Most databases destroy history every time you UPDATE a row. I built one that doesn't. Here's the architecture behind TensorDB, a Rust bitemporal ledger database with 276ns reads."
tags: [rust, database, sql, opensource]
cover_image: https://raw.githubusercontent.com/tensor-db/TensorDB/main/docs/cover.png
---

Last year, a financial services team I was working with had a nightmare scenario: a regulatory audit required them to prove exactly what their system showed on a specific Tuesday six months ago. Not what the data _currently_ says. What it said _then_.

Their production Postgres had the current state. Their audit table had some breadcrumbs. Their application logs were partially rotated. Reconstructing the answer took two engineers three weeks of forensic archaeology through backups, WAL archives, and prayer.

This is the problem that drove me to build [TensorDB](https://github.com/tensor-db/TensorDB).

---

## The Problem With UPDATE

Here's what most databases do when you update a row:

```sql
UPDATE accounts SET balance = 5000 WHERE id = 1;
```

The old value is gone. Destroyed. Overwritten. If you need history, you build it yourself — trigger-based audit tables, event sourcing patterns, CDC pipelines feeding into a data lake. You end up with a Rube Goldberg machine of infrastructure just to answer _"what was this value last week?"_

**Bitemporal databases solve this at the storage layer.** Every write is an immutable fact. Nothing is ever overwritten or deleted. The database tracks two independent timelines for every record:

| Timeline | What it tracks | Example question |
|----------|---------------|-----------------|
| **System time** | When the database _recorded_ this fact | "What did our system show last Tuesday?" |
| **Business time** | When this fact was _true in the real world_ | "What was the contract price on Jan 1?" |

The distinction matters more than you'd think. A bank discovers today that a transaction from January had the wrong amount. With a bitemporal model, you correct the business-time record while preserving the system-time history of what you _previously believed_. Both truths coexist. Auditors can see both.

---

## See It in 30 Seconds

You can have TensorDB running in under a minute:

```bash
pip install tensordb
```

```python
from tensordb import PyDatabase

db = PyDatabase.open("/tmp/demo")

# Create a table and insert data
db.sql("CREATE TABLE accounts (id INT, owner TEXT, balance REAL)")
db.sql("INSERT INTO accounts VALUES (1, 'Alice', 10000)")

# Update the balance
db.sql("UPDATE accounts SET balance = 7500 WHERE id = 1")

# Time-travel: what was Alice's balance BEFORE the update?
rows = db.sql("SELECT * FROM accounts FOR SYSTEM_TIME ALL WHERE id = 1")
print(rows)
# → Both versions: the 10000 AND the 7500, with timestamps
```

That's it. No configuration. No schema migration for audit columns. No background workers. The history is automatic.

Or if you prefer Rust:

```bash
cargo add tensordb
```

```rust
let db = tensordb::Database::open("./mydb")?;

db.sql("CREATE TABLE events (id INT PRIMARY KEY, type TEXT, amount REAL)")?;

db.sql("INSERT INTO events VALUES
    (1, 'deposit', 1000),
    (2, 'withdrawal', 250),
    (3, 'deposit', 500)")?;

// What did the ledger look like at any point in time?
let snapshot = db.sql(
    "SELECT * FROM events AS OF SYSTEM TIME '2026-03-07 12:00:00'"
)?;
```

---

## Why Should You Care?

### It's Fast. Really Fast.

| Operation | TensorDB | SQLite (WAL) | Factor |
|-----------|----------|-------------|--------|
| Point read | **276 ns** | ~400 ns | 1.4x faster |
| Point write | **1.9 us** | ~15 us | **8x faster** |
| Batch insert (10k rows) | **18 ms** | ~35 ms | 2x faster |

These aren't synthetic benchmarks on a tuned cluster. This is single-node, embedded, with full durability guarantees. The write path uses lock-free atomic CAS — no mutexes, no channels, no actor messages on the hot path.

### It Speaks PostgreSQL

```bash
# Start the server
tensordb-server --data-dir ./mydb --port 5433

# Connect with literally anything that speaks Postgres
psql -h localhost -p 5433 -d mydb
```

Your existing tools work — psql, pgAdmin, DBeaver, SQLAlchemy, Prisma, any Postgres driver. You get standard SQL plus temporal queries that Postgres doesn't natively support:

```sql
-- Standard SQL
CREATE TABLE orders (id SERIAL PRIMARY KEY, customer TEXT, total REAL);
INSERT INTO orders (customer, total) VALUES ('acme', 9999) RETURNING id;

-- Temporal queries (the superpower)
SELECT * FROM orders AS OF SYSTEM TIME '2026-01-15';
SELECT * FROM orders FOR SYSTEM_TIME FROM '2026-01-01' TO '2026-03-01';
SELECT * FROM orders VALID AT DATE '2026-02-15';
```

### It Embeds in Your Binary

No daemon process. No Docker container. No ops overhead. One function call:

```rust
let db = Database::open("./path")?;
```

Ship the database _inside_ your application. Ideal for edge deployments, CLI tools, desktop apps, or anywhere a full Postgres deployment is overkill.

### The SQL Surface Is Complete

This isn't a toy query language. It's a full SQL engine with a hand-written recursive descent parser, cost-based query planner, and vectorized execution:

- **DDL/DML:** `CREATE TABLE`, `ALTER TABLE`, `INSERT ... ON CONFLICT` (upsert), `UPDATE ... RETURNING`, `DELETE ... RETURNING`
- **Queries:** JOINs (inner, left, right, full outer, cross), subqueries, CTEs (including `WITH RECURSIVE`), window functions, `GROUP BY`/`HAVING`, `UNION`/`INTERSECT`/`EXCEPT`
- **Types:** `INTEGER`, `REAL`, `TEXT`, `BOOLEAN`, `DATE`, `TIMESTAMP`, `INTERVAL`, `JSON`
- **Functions:** 50+ built-in (string, numeric, date/time, aggregate, window)
- **Advanced:** foreign keys, materialized views, triggers, user-defined functions, generated columns, JSON operators (`->`, `->>`, `@>`)

And the error messages are actually helpful:

```
ERROR T2001: Table "ordres" not found. Did you mean "orders"?
```

---

## How It Works Under the Hood

For those who like to understand the machinery.

### Immutable Key Encoding

Every record gets this internal key:

```
user_key || 0x00 || commit_ts (8B big-endian) || kind (1B)
```

The `user_key` prefix means prefix scans retrieve all versions. Big-endian timestamps give chronological ordering for free. The `kind` byte distinguishes puts from tombstones. Updates don't modify anything — they append new facts with higher timestamps.

### LSM Storage Stack

```
Write ─→ WAL (CRC-framed) ─→ Memtable (BTreeMap)
                                     │ flush
                                     ▼
                            L0 SSTables (sorted)
                                     │ compaction
                                     ▼
                       L1 → L2 → ... → L6
                  (LZ4 for L0-L2, Zstd for L3+)
                  (bloom filters, block cache)
```

**Lock-free writes:** `AtomicU64::compare_exchange` claims a commit timestamp, then writes directly to memtable. No locks on the hot path.

**Direct reads:** `ShardReadHandle` with `parking_lot::RwLock` bypasses shard actors entirely. This is how reads hit 276ns.

**Batched durability:** A `DurabilityThread` coalesces WAL fsyncs across shards on a 1ms interval. Individual writes don't pay fsync cost.

### Cost-Based Query Planner

The planner evaluates plan variants — `PointLookup`, `IndexScan`, `FullScan`, `HashJoin` — using table statistics. A learned cost model tracks actual vs. estimated cardinalities and adjusts its estimates from observed query performance.

---

## Production-Ready Features

Things you'll need when you go beyond prototyping:

**Security:** RBAC with users/roles/permissions, row-level security policies, mTLS on pgwire, column-level AES-256-GCM encryption, encryption key rotation without downtime.

**Audit:** SHA-256 hash-chained audit log. Every DDL and DML event is recorded in a tamper-evident chain. Run `VERIFY AUDIT LOG` to cryptographically verify integrity.

**GDPR:** `FORGET KEY 'user:42'` creates a cryptographic tombstone across all versions — satisfying right-to-erasure while preserving audit log structure.

**Observability:** 8 diagnostic SQL commands — `SHOW STATS`, `SHOW SLOW QUERIES`, `SHOW ACTIVE QUERIES`, `SHOW STORAGE`, `SHOW COMPACTION STATUS`, `SHOW WAL STATUS`, `SHOW AUDIT LOG`, `SHOW PLAN GUIDES`. Plus a health HTTP endpoint.

**Specialized engines:** Full-text search (BM25), time-series (bucketing, gap fill, LOCF, interpolation), vector search (HNSW + IVF-PQ), event sourcing, graph queries.

---

## The Honest Comparison

**Use Postgres** if you need a battle-tested, general-purpose OLTP database with 30 years of production hardening and a massive extension ecosystem.

**Try TensorDB** if:
- Bitemporality is your _primary requirement_, not an afterthought bolted on with triggers
- You want to embed the database directly in your application
- You need structurally append-only storage for compliance (not just "we log changes")
- Sub-microsecond embedded reads matter to you

TensorDB is younger software. It doesn't have Postgres's ecosystem depth. But for the specific problem it solves — immutable, bitemporal, embedded storage with full SQL — it's purpose-built.

---

## Get Started in 60 Seconds

Pick your language:

```bash
# Rust — embed in your binary
cargo add tensordb

# Python — pip install and go
pip install tensordb

# Any language — connect via PostgreSQL protocol
cargo install tensordb-server
tensordb-server --data-dir ./mydb --port 5433
# Then: psql -h localhost -p 5433
```

**Links:**
- [GitHub](https://github.com/tensor-db/TensorDB) — star it if you find it useful
- [Documentation](https://tensor-db.github.io/TensorDB/) — quickstart, SQL reference, architecture guide
- [PyPI](https://pypi.org/project/tensordb/) — `pip install tensordb`
- [crates.io](https://crates.io/crates/tensordb) — `cargo add tensordb`

---

If you're building financial systems, compliance infrastructure, audit trails, healthcare records, or anything where _the history of data matters as much as the current state_ — give it a try and tell me what you think. I read every issue and discussion on GitHub.

And if it breaks, file a bug. That's how it gets better.
