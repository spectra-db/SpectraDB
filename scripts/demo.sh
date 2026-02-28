#!/usr/bin/env bash
# ── TensorDB Live Demo ──────────────────────────────────────────────────
# Usage:  bash scripts/demo.sh [--quick] [--help]
#   --quick   skip the benchmark scene
#   --help    show usage and exit
set -euo pipefail

# ── Flags ────────────────────────────────────────────────────────────────
QUICK=false
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=true ;;
    --help|-h)
      echo "Usage: bash scripts/demo.sh [--quick] [--help]"
      echo "  --quick   skip the benchmark scene"
      exit 0
      ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

# ── Colors ───────────────────────────────────────────────────────────────
BOLD=$'\033[1m'
CYAN=$'\033[36m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
DIM=$'\033[2m'
RESET=$'\033[0m'

# ── Temp database dirs (auto-cleanup) ───────────────────────────────────
DB_DIR=$(mktemp -d /tmp/tensordb-demo-XXXXXX)
PITR_DIR=$(mktemp -d /tmp/tensordb-pitr-XXXXXX)
trap 'rm -rf "$DB_DIR" "$PITR_DIR"' EXIT
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLI="$ROOT/target/release/tensordb-cli"

# ── Helpers ──────────────────────────────────────────────────────────────
scene() {
  echo ""
  echo "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  echo "${BOLD}${CYAN}  $1${RESET}"
  echo "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  echo ""
}

narrate() {
  echo "${DIM}  $1${RESET}"
}

run_sql() {
  local query="$1"
  echo "${YELLOW}  > ${query}${RESET}"
  "$CLI" --path "$DB_DIR" sql "$query" 2>&1 | sed 's/^/    /'
  echo ""
}

run_sql_at() {
  local db="$1" query="$2"
  echo "${YELLOW}  > ${query}${RESET}"
  "$CLI" --path "$db" sql "$query" 2>&1 | sed 's/^/    /'
  echo ""
}

pause() {
  sleep 0.3
}

# ── Build ────────────────────────────────────────────────────────────────
scene "Building TensorDB CLI"
narrate "Compiling tensordb-cli in release mode..."
export BINDGEN_EXTRA_CLANG_ARGS="${BINDGEN_EXTRA_CLANG_ARGS:---include-directory=/usr/lib/gcc/aarch64-linux-gnu/13/include}"
(cd "$ROOT" && cargo build --release -p tensordb-cli 2>&1 | tail -1)

if [[ ! -x "$CLI" ]]; then
  echo "${BOLD}${YELLOW}ERROR: Failed to build tensordb-cli${RESET}"
  exit 1
fi
echo "${GREEN}  Ready.${RESET}"

# ═════════════════════════════════════════════════════════════════════════
# Scene 1: One Database Replaces Five
# ═════════════════════════════════════════════════════════════════════════
scene "Scene 1: One Database Replaces Five"
narrate "Relational, full-text search, and time-series — all in one engine."
echo ""

# ── Relational ───────────────────────────────────────────────────────────
narrate "▸ Relational tables with typed schemas"
pause

run_sql "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, dept TEXT, salary REAL);"
run_sql "INSERT INTO employees (id, name, dept, salary) VALUES (1, 'Alice',   'engineering', 130000);"
run_sql "INSERT INTO employees (id, name, dept, salary) VALUES (2, 'Bob',     'engineering', 120000);"
run_sql "INSERT INTO employees (id, name, dept, salary) VALUES (3, 'Charlie', 'marketing',   95000);"
run_sql "INSERT INTO employees (id, name, dept, salary) VALUES (4, 'Diana',   'marketing',   105000);"
run_sql "INSERT INTO employees (id, name, dept, salary) VALUES (5, 'Eve',     'sales',       110000);"

narrate "Aggregation — average salary by department:"
run_sql "SELECT dept, COUNT(*) AS headcount, AVG(salary) AS avg_salary FROM employees GROUP BY dept ORDER BY dept ASC;"

narrate "Window function — rank employees by salary within each department:"
run_sql "SELECT name, dept, salary, RANK() OVER (PARTITION BY dept ORDER BY salary DESC) AS dept_rank FROM employees ORDER BY dept ASC, salary DESC;"

# ── Full-Text Search ────────────────────────────────────────────────────
echo ""
narrate "▸ Built-in full-text search"
pause

run_sql "CREATE TABLE articles (id INTEGER PRIMARY KEY, title TEXT, body TEXT);"
run_sql "CREATE FULLTEXT INDEX idx_articles ON articles (title, body);"

run_sql "INSERT INTO articles (id, title, body) VALUES (1, 'Getting Started with Rust', 'Rust is a systems programming language focused on safety and performance');"
run_sql "INSERT INTO articles (id, title, body) VALUES (2, 'Python for Data Science', 'Python excels at data analysis and machine learning workflows');"
run_sql "INSERT INTO articles (id, title, body) VALUES (3, 'Database Internals', 'How modern databases use LSM trees and write-ahead logs for durability');"

narrate "Search for 'rust' across title and body:"
run_sql "SELECT title FROM articles WHERE MATCH(title, 'rust');"

narrate "Highlight matching terms:"
run_sql "SELECT HIGHLIGHT(body, 'programming') FROM articles;"

# ── Time-Series ──────────────────────────────────────────────────────────
echo ""
narrate "▸ Native time-series with bucketing"
pause

run_sql "CREATE TIMESERIES TABLE cpu_metrics (id INTEGER PRIMARY KEY, ts INTEGER, cpu REAL) WITH (bucket_size = '1m');"

run_sql "INSERT INTO cpu_metrics (id, ts, cpu) VALUES (1, 1000, 45.2);"
run_sql "INSERT INTO cpu_metrics (id, ts, cpu) VALUES (2, 1015, 52.1);"
run_sql "INSERT INTO cpu_metrics (id, ts, cpu) VALUES (3, 1030, 48.7);"
run_sql "INSERT INTO cpu_metrics (id, ts, cpu) VALUES (4, 1060, 71.3);"
run_sql "INSERT INTO cpu_metrics (id, ts, cpu) VALUES (5, 1075, 68.9);"
run_sql "INSERT INTO cpu_metrics (id, ts, cpu) VALUES (6, 1090, 65.0);"

narrate "Aggregate into 1-minute buckets:"
run_sql "SELECT time_bucket('1m', ts) AS bucket, AVG(cpu) AS avg_cpu, FIRST(cpu, ts) AS first_cpu, LAST(cpu, ts) AS last_cpu FROM cpu_metrics GROUP BY time_bucket('1m', ts) ORDER BY time_bucket('1m', ts) ASC;"

# ═════════════════════════════════════════════════════════════════════════
# Scene 2: Time Machine (Epoch-Based PITR)
# ═════════════════════════════════════════════════════════════════════════
scene "Scene 2: Time Machine — Point-in-Time Recovery"
narrate "Every transaction gets an epoch. You can query any past state."
echo ""

narrate "Three transactions: initial stock → restock → fire sale"
narrate "(all in one session so epoch counters are preserved)"
pause

# Run all writes in a single CLI invocation on a clean database so epochs
# are sequential and shard commit counters are properly synchronized.
# execute_sql splits on ; and uses one SqlSession — transactions work.
run_sql_at "$PITR_DIR" "CREATE TABLE inventory (id TEXT PRIMARY KEY, item TEXT, qty INTEGER); BEGIN; INSERT INTO inventory (id, item, qty) VALUES ('sku-1', 'Keyboard', 50); INSERT INTO inventory (id, item, qty) VALUES ('sku-2', 'Mouse',    100); INSERT INTO inventory (id, item, qty) VALUES ('sku-3', 'Monitor',  25); COMMIT; BEGIN; UPDATE inventory SET doc = '{\"item\":\"Keyboard\",\"qty\":150}' WHERE pk = 'sku-1'; UPDATE inventory SET doc = '{\"item\":\"Mouse\",\"qty\":200}' WHERE pk = 'sku-2'; COMMIT; BEGIN; UPDATE inventory SET doc = '{\"item\":\"Keyboard\",\"qty\":5}' WHERE pk = 'sku-1'; UPDATE inventory SET doc = '{\"item\":\"Mouse\",\"qty\":3}' WHERE pk = 'sku-2'; UPDATE inventory SET doc = '{\"item\":\"Monitor\",\"qty\":1}' WHERE pk = 'sku-3'; COMMIT;"

narrate "Current state (after fire sale):"
run_sql_at "$PITR_DIR" "SELECT item, qty FROM inventory ORDER BY item ASC;"

narrate "Travel back to epoch 3 (after restock, before fire sale):"
run_sql_at "$PITR_DIR" "SELECT item, qty FROM inventory AS OF EPOCH 3 ORDER BY item ASC;"

narrate "Travel back to epoch 2 (initial stock only):"
run_sql_at "$PITR_DIR" "SELECT item, qty FROM inventory AS OF EPOCH 2 ORDER BY item ASC;"

# ═════════════════════════════════════════════════════════════════════════
# Scene 3: Bitemporal Queries — Application Time
# ═════════════════════════════════════════════════════════════════════════
scene "Scene 3: Bitemporal Queries — Application Time"
narrate "TensorDB is bitemporal: system time (shown in Scene 2) tracks WHEN data was"
narrate "recorded; application time tracks WHEN data is valid in the business domain."
echo ""

narrate "Create a table, then insert records with explicit validity windows:"
pause

run_sql "CREATE TABLE revenue (pk TEXT PRIMARY KEY);"

echo "${YELLOW}  > put table/revenue/q1 '{\"quarter\":\"Q1\",\"revenue\":100000}' --valid-from 100 --valid-to 200${RESET}"
"$CLI" --path "$DB_DIR" put "table/revenue/q1" '{"quarter":"Q1","revenue":100000}' --valid-from 100 --valid-to 200 2>&1 | sed 's/^/    /'
echo ""
echo "${YELLOW}  > put table/revenue/q2 '{\"quarter\":\"Q2\",\"revenue\":120000}' --valid-from 200 --valid-to 300${RESET}"
"$CLI" --path "$DB_DIR" put "table/revenue/q2" '{"quarter":"Q2","revenue":120000}' --valid-from 200 --valid-to 300 2>&1 | sed 's/^/    /'
echo ""
echo "${YELLOW}  > put table/revenue/q3 '{\"quarter\":\"Q3\",\"revenue\":95000}' --valid-from 300 --valid-to 400${RESET}"
"$CLI" --path "$DB_DIR" put "table/revenue/q3" '{"quarter":"Q3","revenue":95000}' --valid-from 300 --valid-to 400 2>&1 | sed 's/^/    /'
echo ""

narrate "Query records valid at application-time 150 (only Q1):"
run_sql "SELECT doc FROM revenue FOR APPLICATION_TIME AS OF 150;"

narrate "Query records valid at application-time 250 (only Q2):"
run_sql "SELECT doc FROM revenue FOR APPLICATION_TIME AS OF 250;"

narrate "Query records valid at application-time 350 (only Q3):"
run_sql "SELECT doc FROM revenue FOR APPLICATION_TIME AS OF 350;"

# ═════════════════════════════════════════════════════════════════════════
# Scene 4: Real Transactions
# ═════════════════════════════════════════════════════════════════════════
scene "Scene 4: Real Transactions — BEGIN / COMMIT / ROLLBACK / SAVEPOINT"
narrate "ACID transactions with savepoint support."
echo ""

run_sql "CREATE TABLE ledger (id TEXT PRIMARY KEY, description TEXT, amount REAL);"

narrate "Transaction 1: COMMIT — persists"
run_sql "BEGIN; INSERT INTO ledger (id, description, amount) VALUES ('txn-1', 'Initial deposit', 1000.0); INSERT INTO ledger (id, description, amount) VALUES ('txn-2', 'Wire transfer', 500.0); COMMIT;"

narrate "Transaction 2: ROLLBACK — discarded"
run_sql "BEGIN; INSERT INTO ledger (id, description, amount) VALUES ('txn-3', 'Accidental charge', -9999.0); ROLLBACK;"

narrate "Verify: only committed rows exist"
run_sql "SELECT id, description, amount FROM ledger ORDER BY id ASC;"

narrate "Transaction 3: SAVEPOINT — partial rollback"
run_sql "BEGIN; INSERT INTO ledger (id, description, amount) VALUES ('txn-4', 'Bonus payment', 250.0); SAVEPOINT sp1; INSERT INTO ledger (id, description, amount) VALUES ('txn-5', 'Disputed refund', -100.0); ROLLBACK TO sp1; COMMIT;"

narrate "Verify: txn-4 kept, txn-5 undone"
run_sql "SELECT id, description, amount FROM ledger ORDER BY id ASC;"

# ═════════════════════════════════════════════════════════════════════════
# Scene 5: Talk to Your Database (LLM)
# ═════════════════════════════════════════════════════════════════════════
scene "Scene 5: Talk to Your Database — Natural Language → SQL"
narrate "TensorDB embeds a Qwen3-0.6B LLM for natural-language queries."
echo ""

# Search common model locations
MODEL_FILE=""
for candidate in \
  "$DB_DIR/.local/models/Qwen3-0.6B-Q8_0.gguf" \
  "$ROOT/.local/models/Qwen3-0.6B-Q8_0.gguf" \
  "$HOME/.local/share/tensordb/models/Qwen3-0.6B-Q8_0.gguf" \
  "$HOME/models/Qwen3-0.6B-Q8_0.gguf"; do
  if [[ -f "$candidate" ]]; then
    MODEL_FILE="$candidate"
    break
  fi
done

if [[ -n "$MODEL_FILE" ]]; then
  narrate "Model found: $MODEL_FILE"
  narrate "Asking: 'how many employees are in engineering?'"
  run_sql "ASK 'how many employees are in engineering?';"
else
  narrate "LLM model not found — showing what it looks like:"
  echo ""
  echo "${YELLOW}  > ASK 'how many employees are in engineering?';${RESET}"
  echo "${DIM}    Generated SQL: SELECT COUNT(*) FROM employees WHERE dept = 'engineering'${RESET}"
  echo "${DIM}    ┌───────────┐${RESET}"
  echo "${DIM}    │ COUNT(*)  │${RESET}"
  echo "${DIM}    ├───────────┤${RESET}"
  echo "${DIM}    │ 2         │${RESET}"
  echo "${DIM}    └───────────┘${RESET}"
  echo ""
  narrate "To enable: download Qwen3-0.6B-Q8_0.gguf into <db_root>/.local/models/"
fi

# ═════════════════════════════════════════════════════════════════════════
# Scene 6: Performance (skipped with --quick)
# ═════════════════════════════════════════════════════════════════════════
if [[ "$QUICK" == false ]]; then
  scene "Scene 6: Performance"
  narrate "Running 10,000 writes + 5,000 reads against TensorDB..."
  echo ""

  echo "${YELLOW}  > tensordb-cli bench --write-ops 10000 --read-ops 5000${RESET}"
  "$CLI" --path "$DB_DIR" bench --write-ops 10000 --read-ops 5000 2>&1 | sed 's/^/    /'
  echo ""
  narrate "For context: point reads ~276ns (4x faster than SQLite),"
  narrate "fast-path writes ~1.9µs (20x faster than SQLite)."
else
  narrate "(Benchmark skipped — rerun without --quick to see performance numbers)"
fi

# ═════════════════════════════════════════════════════════════════════════
# Epilogue
# ═════════════════════════════════════════════════════════════════════════
echo ""
echo "${BOLD}${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo "${BOLD}${GREEN}  Demo Complete${RESET}"
echo "${BOLD}${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
narrate "What you just saw:"
narrate "  1. Relational + FTS + Time-Series in one engine"
narrate "  2. Epoch-based point-in-time recovery (time travel)"
narrate "  3. Bitemporal audit trail (system time + application time)"
narrate "  4. ACID transactions with savepoints"
narrate "  5. Natural-language SQL via embedded LLM"
if [[ "$QUICK" == false ]]; then
  narrate "  6. Sub-microsecond read performance"
fi
echo ""
narrate "Learn more: https://github.com/tensor-db/tensorDB"
narrate "Temp database cleaned up on exit."
echo ""
