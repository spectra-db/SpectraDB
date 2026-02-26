# Changelog

All notable changes to SpectraDB will be documented in this file.

## [Unreleased]

### Added
- Tier-0 in-core AI runtime that consumes change feeds, synthesizes insights, and persists immutable internal AI facts.
- CLI AI operations: `.ai_status` and `.ai <key> [limit]`.
- CLI AI operator commands: `.ai_timeline`, `.ai_correlate`, `.ai_explain`.
- AI runtime controls for batching: `ai_batch_window_ms` and `ai_batch_max_events`.
- CI AI overhead gate script: `scripts/ai_overhead_gate.sh`.
- Release AI overhead report script: `scripts/release_ai_report.sh`.
- Executable quickstart docs coverage: `tests/readme_examples.rs`.

### Changed
- AI integration is now native in-process only (no external model server/subprocess path).
- Removed backend selection from public AI runtime configuration surface.
- Updated docs, roadmap, and test plans to reflect current SQL surface and AI architecture.

## [0.1.0] - 2026-02-25

### Added
- Durable append-only fact ledger with WAL and CRC-framed records.
- MVCC snapshot reads (`AS OF <commit_ts>`).
- Bitemporal filtering (`VALID AT <valid_ts>`).
- Sharded single-writer execution model.
- LSM-style SSTables with bloom filters, block index, and mmap reads.
- RelationalFacet SQL subset: CREATE TABLE/VIEW/INDEX, ALTER TABLE ADD COLUMN, INSERT, SELECT, DROP, SHOW TABLES, DESCRIBE, EXPLAIN, transactions (BEGIN/COMMIT/ROLLBACK).
- Interactive CLI shell with TAB autocomplete, persistent history, and output modes (table/line/json).
- Optional C++ acceleration behind `--features native` via `cxx`.
- Benchmark harness with configurable workload matrix.
- Overnight burn-in script for reliability testing.
