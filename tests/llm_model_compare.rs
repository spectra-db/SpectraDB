/// Model comparison: runs complex NL->SQL questions against whichever model is set
/// via LLM_MODEL_PATH, focusing on TensorDB-specific features.
///
/// Run both models back-to-back:
///   LLM_MODEL_PATH=~/.tensordb/models/prem-1B-SQL.Q8_0.gguf \
///     cargo test --test llm_model_compare -- --test-threads=1 --nocapture 2>&1 | tee prem_results.txt
///
///   LLM_MODEL_PATH=~/.tensordb/models/Qwen3-0.6B-Q8_0.gguf \
///     cargo test --test llm_model_compare -- --test-threads=1 --nocapture 2>&1 | tee qwen_results.txt
#[cfg(feature = "llm")]
mod tests {
    use tensordb_core::config::Config;
    use tensordb_core::engine::db::Database;

    fn model_available() -> bool {
        std::env::var("LLM_MODEL_PATH")
            .ok()
            .map(|p| std::path::Path::new(&p).exists())
            .unwrap_or(false)
    }

    fn model_name() -> String {
        std::env::var("LLM_MODEL_PATH")
            .ok()
            .and_then(|p| {
                std::path::Path::new(&p)
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
            })
            .unwrap_or_else(|| "unknown".to_string())
    }

    fn open_test_db(name: &str) -> Option<Database> {
        if !model_available() {
            eprintln!("Skipping: no LLM model file found (set LLM_MODEL_PATH)");
            return None;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        let _ = std::fs::create_dir_all(&path);

        let config = Config {
            shard_count: 1,
            ai_auto_insights: false,
            llm_model_path: std::env::var("LLM_MODEL_PATH").ok(),
            ..Config::default()
        };

        let db = Database::open(path, config).expect("open db");

        // Rich multi-table schema
        db.sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT, role TEXT, created_at INTEGER);").unwrap();
        db.sql("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, product TEXT, amount REAL, status TEXT, ordered_at INTEGER);").unwrap();
        db.sql("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL, category TEXT, stock INTEGER);").unwrap();
        db.sql("CREATE TABLE events (id INTEGER PRIMARY KEY, user_id INTEGER, event_type TEXT, payload TEXT, ts INTEGER);").unwrap();
        db.sql("CREATE TABLE metrics (id INTEGER PRIMARY KEY, sensor_id INTEGER, value REAL, recorded_at INTEGER);").unwrap();

        // Populate users
        for (id, name, email, role, ts) in [
            (1, "alice", "alice@co.com", "admin", 1700000000),
            (2, "bob", "bob@co.com", "user", 1700100000),
            (3, "carol", "carol@co.com", "admin", 1700200000),
            (4, "dave", "dave@co.com", "user", 1700300000),
            (5, "eve", "eve@co.com", "moderator", 1700400000),
            (6, "frank", "frank@co.com", "user", 1700500000),
        ] {
            db.sql(&format!(
                "INSERT INTO users (id, name, email, role, created_at) VALUES ({id}, '{name}', '{email}', '{role}', {ts});"
            )).unwrap();
        }

        // Populate orders
        for (id, uid, product, amount, status, ts) in [
            (101, 1, "Laptop", 999.99, "shipped", 1700500000),
            (102, 2, "Mouse", 29.99, "delivered", 1700600000),
            (103, 1, "Keyboard", 79.99, "delivered", 1700700000),
            (104, 3, "Monitor", 449.99, "pending", 1700800000),
            (105, 4, "Webcam", 59.99, "shipped", 1700900000),
            (106, 2, "Headset", 149.99, "cancelled", 1701000000),
            (107, 5, "Laptop", 999.99, "delivered", 1701100000),
            (108, 6, "Mouse", 29.99, "pending", 1701200000),
            (109, 1, "Monitor", 449.99, "shipped", 1701300000),
            (110, 3, "Keyboard", 79.99, "delivered", 1701400000),
        ] {
            db.sql(&format!(
                "INSERT INTO orders (id, user_id, product, amount, status, ordered_at) VALUES ({id}, {uid}, '{product}', {amount}, '{status}', {ts});"
            )).unwrap();
        }

        // Populate products
        for (id, name, price, category, stock) in [
            (1, "Laptop", 999.99, "electronics", 50),
            (2, "Mouse", 29.99, "accessories", 500),
            (3, "Keyboard", 79.99, "accessories", 200),
            (4, "Monitor", 449.99, "electronics", 75),
            (5, "Webcam", 59.99, "accessories", 150),
            (6, "Headset", 149.99, "audio", 100),
        ] {
            db.sql(&format!(
                "INSERT INTO products (id, name, price, category, stock) VALUES ({id}, '{name}', {price}, '{category}', {stock});"
            )).unwrap();
        }

        // Populate events
        for (id, uid, etype, payload, ts) in [
            (1, 1, "login", "ip=10.0.0.1", 1700500000),
            (2, 2, "login", "ip=10.0.0.2", 1700600000),
            (3, 1, "purchase", "order=101", 1700500000),
            (4, 3, "login", "ip=10.0.0.3", 1700800000),
            (5, 1, "logout", "session=abc", 1700900000),
        ] {
            db.sql(&format!(
                "INSERT INTO events (id, user_id, event_type, payload, ts) VALUES ({id}, {uid}, '{etype}', '{payload}', {ts});"
            )).unwrap();
        }

        // Populate metrics
        for (id, sensor, value, ts) in [
            (1, 1, 23.5, 1700000000),
            (2, 1, 24.1, 1700003600),
            (3, 2, 45.0, 1700000000),
            (4, 2, 46.2, 1700003600),
            (5, 1, 22.9, 1700007200),
        ] {
            db.sql(&format!(
                "INSERT INTO metrics (id, sensor_id, value, recorded_at) VALUES ({id}, {sensor}, {value}, {ts});"
            )).unwrap();
        }

        std::mem::forget(dir);
        Some(db)
    }

    fn ask_and_report(
        db: &Database,
        question: &str,
    ) -> (String, std::result::Result<String, String>) {
        eprintln!("\n----------------------------------------------------------------------");
        eprintln!("Q: {question}");

        match db.ask_sql(question) {
            Ok(sql) => {
                eprintln!("  SQL: {sql}");
                match db.sql(&sql) {
                    Ok(result) => {
                        let result_str = format!("{result:?}");
                        let display = if result_str.len() > 300 {
                            format!("{}...", &result_str[..300])
                        } else {
                            result_str.clone()
                        };
                        eprintln!("  Result: {display}");
                        (sql, Ok(result_str))
                    }
                    Err(e) => {
                        eprintln!("  EXEC ERROR: {e}");
                        (sql, Err(e.to_string()))
                    }
                }
            }
            Err(e) => {
                eprintln!("  GEN ERROR: {e}");
                (String::new(), Err(e.to_string()))
            }
        }
    }

    fn sql_contains(sql: &str, keywords: &[&str]) -> bool {
        let upper = sql.to_uppercase();
        keywords.iter().all(|kw| upper.contains(&kw.to_uppercase()))
    }

    #[test]
    fn complex_scorecard() {
        let Some(db) = open_test_db("complex_scorecard") else {
            return;
        };

        let model = model_name();

        struct Q {
            tier: &'static str,
            question: &'static str,
            required_keywords: Vec<&'static str>,
            desc: &'static str,
        }

        let questions = vec![
            // ── Basics (sanity check) ────────────────────────────────────
            Q {
                tier: "BASIC",
                question: "How many users are there?",
                required_keywords: vec!["SELECT", "COUNT"],
                desc: "Simple COUNT",
            },
            Q {
                tier: "BASIC",
                question: "Show all products sorted by price descending",
                required_keywords: vec!["SELECT", "products", "ORDER"],
                desc: "ORDER BY DESC",
            },
            // ── Multi-table JOINs ────────────────────────────────────────
            Q {
                tier: "JOIN",
                question: "Show each user's name alongside the products they ordered and each product's category",
                required_keywords: vec!["SELECT", "JOIN"],
                desc: "3-table JOIN (users-orders-products)",
            },
            Q {
                tier: "JOIN",
                question: "For each order, show the user name, product name, product category, and order status",
                required_keywords: vec!["SELECT", "JOIN"],
                desc: "Multi-table JOIN with all fields",
            },
            // ── Aggregation + GROUP BY + HAVING ──────────────────────────
            Q {
                tier: "AGG",
                question: "What is the average order amount per user? Only show users with average above 100",
                required_keywords: vec!["SELECT", "AVG", "GROUP BY"],
                desc: "AVG + GROUP BY + HAVING",
            },
            Q {
                tier: "AGG",
                question: "How many orders are in each status? Sort by count descending",
                required_keywords: vec!["SELECT", "COUNT", "GROUP BY", "ORDER"],
                desc: "COUNT + GROUP BY + ORDER BY",
            },
            Q {
                tier: "AGG",
                question: "What is the total revenue per product category?",
                required_keywords: vec!["SELECT", "SUM"],
                desc: "SUM with category grouping",
            },
            // ── Subqueries ───────────────────────────────────────────────
            Q {
                tier: "SUBQUERY",
                question: "Find users who have spent more than the average order amount",
                required_keywords: vec!["SELECT", "users"],
                desc: "Subquery with AVG comparison",
            },
            Q {
                tier: "SUBQUERY",
                question: "Which products have never been ordered?",
                required_keywords: vec!["SELECT", "products"],
                desc: "NOT IN / NOT EXISTS subquery",
            },
            // ── CASE expressions ─────────────────────────────────────────
            Q {
                tier: "CASE",
                question: "Classify each order as 'small' if amount < 50, 'medium' if 50-200, 'large' if > 200",
                required_keywords: vec!["SELECT", "CASE"],
                desc: "Multi-bucket CASE WHEN",
            },
            // ── CTE (WITH) ──────────────────────────────────────────────
            Q {
                tier: "CTE",
                question: "Using a CTE, compute each user's total spend, then show users who spent over 500 along with their names",
                required_keywords: vec!["SELECT"],
                desc: "CTE + aggregation + filter",
            },
            // ── TensorDB temporal ────────────────────────────────────────
            Q {
                tier: "TEMPORAL",
                question: "Show the users table as it was at commit timestamp 2",
                required_keywords: vec!["SELECT", "AS OF"],
                desc: "Time-travel AS OF",
            },
            Q {
                tier: "TEMPORAL",
                question: "What did the orders table look like at epoch 5?",
                required_keywords: vec!["SELECT"],
                desc: "Epoch-based time travel",
            },
            // ── Schema introspection ─────────────────────────────────────
            Q {
                tier: "META",
                question: "What columns does the events table have?",
                required_keywords: vec!["DESCRIBE"],
                desc: "DESCRIBE table",
            },
            Q {
                tier: "META",
                question: "List all tables in the database",
                required_keywords: vec!["SHOW"],
                desc: "SHOW TABLES",
            },
            // ── Complex multi-step ───────────────────────────────────────
            Q {
                tier: "COMPLEX",
                question: "Show the top 3 users by total order spend, including their name and total amount",
                required_keywords: vec!["SELECT", "ORDER", "LIMIT"],
                desc: "Top-N with aggregation + JOIN",
            },
            Q {
                tier: "COMPLEX",
                question: "For each product category, find the most expensive product name and its price",
                required_keywords: vec!["SELECT", "products"],
                desc: "Per-group MAX with name",
            },
            Q {
                tier: "COMPLEX",
                question: "Show users who placed orders for both 'Laptop' and 'Keyboard'",
                required_keywords: vec!["SELECT"],
                desc: "Set intersection query",
            },
            Q {
                tier: "COMPLEX",
                question: "Count how many distinct products each user has ordered, and show only users with 2 or more distinct products",
                required_keywords: vec!["SELECT", "COUNT", "GROUP BY"],
                desc: "COUNT DISTINCT + HAVING",
            },
            // ── Ambiguous / hard NL ──────────────────────────────────────
            Q {
                tier: "AMBIGUOUS",
                question: "Who is the biggest spender?",
                required_keywords: vec!["SELECT"],
                desc: "Ambiguous phrasing",
            },
            Q {
                tier: "AMBIGUOUS",
                question: "What's trending? Show the most ordered products",
                required_keywords: vec!["SELECT"],
                desc: "Informal language",
            },
        ];

        let mut results: Vec<(&str, &str, bool, bool, String)> = Vec::new();
        let mut pass = 0;
        let mut sql_ok_exec_fail = 0;
        let mut fail = 0;

        // Report kernel dispatch info for reproducibility
        {
            let features = tensordb_core::ai::kernels::cpu_features();
            let tier = tensordb_core::ai::kernels::best_int_kernel();
            eprintln!("KERNEL TIER: {} (features: {:?})", tier, features);
        }

        eprintln!("\n======================================================================");
        eprintln!("MODEL COMPARISON SCORECARD: {model}");
        eprintln!("======================================================================");

        let start = std::time::Instant::now();

        for q in &questions {
            let q_start = std::time::Instant::now();
            let (sql, result) = ask_and_report(&db, q.question);
            let q_elapsed = q_start.elapsed();
            eprintln!("  TIME: {:.2}s", q_elapsed.as_secs_f64());

            let kw_ok = if sql.is_empty() {
                false
            } else {
                sql_contains(&sql, &q.required_keywords)
            };
            let exec_ok = result.is_ok();

            let status = if kw_ok && exec_ok {
                pass += 1;
                "PASS"
            } else if kw_ok {
                sql_ok_exec_fail += 1;
                "SQL OK, EXEC FAIL"
            } else {
                fail += 1;
                "FAIL"
            };

            results.push((q.tier, q.desc, kw_ok, exec_ok, sql.clone()));
            eprintln!(
                "  [{:>10}] {:>16} | {} — {}",
                q.tier, status, q.desc, q.question
            );
        }

        let elapsed = start.elapsed();

        eprintln!("\n======================================================================");
        eprintln!("MODEL: {model}");
        eprintln!(
            "RESULTS: {pass} passed, {fail} failed, {sql_ok_exec_fail} sql-ok-exec-fail out of {}",
            questions.len()
        );
        eprintln!(
            "TIME: {:.1}s total, {:.1}s per question avg",
            elapsed.as_secs_f64(),
            elapsed.as_secs_f64() / questions.len() as f64
        );
        eprintln!("======================================================================");

        // Breakdown by tier
        let tiers = [
            "BASIC",
            "JOIN",
            "AGG",
            "SUBQUERY",
            "CASE",
            "CTE",
            "TEMPORAL",
            "META",
            "COMPLEX",
            "AMBIGUOUS",
        ];
        for tier in &tiers {
            let tier_results: Vec<_> = results.iter().filter(|r| r.0 == *tier).collect();
            if tier_results.is_empty() {
                continue;
            }
            let tier_pass = tier_results.iter().filter(|r| r.2 && r.3).count();
            eprintln!("  {tier:>10}: {tier_pass}/{} passed", tier_results.len());
        }
        eprintln!("======================================================================\n");
    }
}
