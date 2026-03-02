//! NativeLlmEngine — pure-Rust inference for NL-to-SQL translation.
//!
//! Custom GGUF loader, BPE tokenizer, Qwen2 transformer runtime, and
//! constrained SQL grammar decoder. Pure Rust — no C++ dependencies.
//!
//! Key features:
//! - **Schema cache**: TTL-based caching avoids re-running SHOW TABLES + DESCRIBE.
//! - **Constrained decoding**: SQL grammar decoder biases generation toward valid SQL.
//! - **Table filtering**: Schema context pruned to relevant tables for the 0.6B model.

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use crate::error::{Result, TensorError};

use super::gguf::GgufFile;
use super::sampler::Sampler;
use super::schema_cache::SchemaCache;
use super::sql_grammar::SqlGrammarDecoder;
use super::tokenizer::BpeTokenizer;
use super::transformer::{KvCache, ModelConfig, ScratchBuffers, TransformerModel};

const DEFAULT_MAX_TOKENS: usize = 256;
const DEFAULT_CONTEXT_SIZE: usize = 2048;
const DEFAULT_SCHEMA_CACHE_TTL_SECS: u64 = 60;
const MODEL_FILENAME: &str = "Qwen3-0.6B-Q8_0.gguf";
const MODEL_URL: &str =
    "https://github.com/tensor-db/TensorDB/releases/download/v0.2.0-model/Qwen3-0.6B-Q8_0.gguf";

const SYSTEM_PROMPT: &str = "\
You are a SQL translator. Generate SQL only. /no_think";

struct LoadedModel {
    model: TransformerModel,
    tokenizer: BpeTokenizer,
    grammar: SqlGrammarDecoder,
    config: ModelConfig,
    scratch: ScratchBuffers,
}

pub struct LlmEngine {
    inner: Mutex<Option<LoadedModel>>,
    model_path: PathBuf,
    loaded: AtomicBool,
    max_tokens: usize,
    context_size: usize,
    schema_cache: SchemaCache,
    grammar_constrained: bool,
}

impl LlmEngine {
    pub fn new(model_path: PathBuf) -> Self {
        Self {
            inner: Mutex::new(None),
            model_path,
            loaded: AtomicBool::new(false),
            max_tokens: DEFAULT_MAX_TOKENS,
            context_size: DEFAULT_CONTEXT_SIZE,
            schema_cache: SchemaCache::new(DEFAULT_SCHEMA_CACHE_TTL_SECS),
            grammar_constrained: true,
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_context_size(mut self, context_size: usize) -> Self {
        self.context_size = context_size;
        self
    }

    pub fn with_schema_cache_ttl(self, ttl_secs: u64) -> Self {
        // Replace the schema cache with new TTL (must rebuild since SchemaCache isn't Clone)
        Self {
            schema_cache: SchemaCache::new(ttl_secs),
            ..self
        }
    }

    pub fn with_grammar_constrained(mut self, enabled: bool) -> Self {
        self.grammar_constrained = enabled;
        self
    }

    fn ensure_loaded(&self) -> Result<()> {
        if self.loaded.load(Ordering::Acquire) {
            return Ok(());
        }

        let mut guard = self.inner.lock();
        // Double-check after acquiring lock
        if self.loaded.load(Ordering::Acquire) {
            return Ok(());
        }

        // Auto-download if model file doesn't exist
        if !self.model_path.exists() {
            Self::download_model(&self.model_path)?;
        }

        // Load GGUF file
        let gguf = GgufFile::open(&self.model_path)?;

        // Extract model config
        let config = ModelConfig::from_gguf(&gguf)?;

        // Load tokenizer
        let tokenizer = BpeTokenizer::from_gguf(&gguf)?;

        // Build grammar decoder
        let grammar = SqlGrammarDecoder::new(&tokenizer, self.grammar_constrained);

        // Load transformer model
        let model = TransformerModel::from_gguf(&gguf, &config)?;

        // Pre-allocate scratch buffers for zero-alloc generation
        let scratch = ScratchBuffers::new(&config);

        *guard = Some(LoadedModel {
            model,
            tokenizer,
            grammar,
            config,
            scratch,
        });
        self.loaded.store(true, Ordering::Release);
        Ok(())
    }

    fn download_model(dest: &std::path::Path) -> Result<()> {
        eprintln!("Downloading TensorDB language model ({MODEL_FILENAME}, ~604 MB)...");
        eprintln!("  From: {MODEL_URL}");
        eprintln!("  To:   {}", dest.display());

        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                TensorError::LlmError(format!(
                    "failed to create model directory {}: {e}",
                    parent.display()
                ))
            })?;
        }

        // Download to a temp file first, then rename (atomic-ish)
        let tmp_path = dest.with_extension("gguf.download");

        let resp = ureq::get(MODEL_URL)
            .call()
            .map_err(|e| TensorError::LlmError(format!("failed to download model: {e}")))?;

        let total: u64 = resp
            .header("content-length")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let mut reader = resp.into_reader();
        let mut file = std::fs::File::create(&tmp_path).map_err(|e| {
            TensorError::LlmError(format!("failed to create {}: {e}", tmp_path.display()))
        })?;

        let mut buf = vec![0u8; 1024 * 1024]; // 1 MB buffer
        let mut downloaded: u64 = 0;
        let mut last_pct: u64 = 0;

        loop {
            let n = reader
                .read(&mut buf)
                .map_err(|e| TensorError::LlmError(format!("download read error: {e}")))?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n])
                .map_err(|e| TensorError::LlmError(format!("write error: {e}")))?;
            downloaded += n as u64;

            if total > 0 {
                let pct = downloaded * 100 / total;
                if pct >= last_pct + 5 {
                    eprint!(
                        "\r  Progress: {pct}% ({} / {} MB)",
                        downloaded / (1024 * 1024),
                        total / (1024 * 1024),
                    );
                    last_pct = pct;
                }
            }
        }
        eprintln!("\r  Progress: 100% — download complete.       ");

        drop(file);
        std::fs::rename(&tmp_path, dest)
            .map_err(|e| TensorError::LlmError(format!("failed to finalize model file: {e}")))?;

        eprintln!("  Model ready.");
        Ok(())
    }

    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        self.ensure_loaded()?;

        let mut guard = self.inner.lock();
        let loaded = guard.as_mut().ok_or(TensorError::LlmNotAvailable)?;

        let tokens = loaded.tokenizer.encode(prompt);
        if tokens.is_empty() {
            return Err(TensorError::LlmError(
                "prompt tokenized to empty sequence".to_string(),
            ));
        }

        let ctx_size = self.context_size;
        let mut kv_cache = KvCache::new(
            loaded.config.n_layers,
            loaded.config.n_kv_heads,
            loaded.config.head_dim,
            ctx_size,
        );

        // Prefill: process all prompt tokens using scratch buffers
        loaded
            .model
            .forward_batch_into(&tokens, 0, &mut kv_cache, &mut loaded.scratch);

        // Generation loop (repetition penalty prevents degenerate loops)
        let mut sampler = Sampler::new(0.0, 1.0, 1.3, 0);
        let mut output_tokens: Vec<u32> = Vec::new();
        let mut pos = tokens.len();

        for _ in 0..max_tokens {
            // Apply grammar constraints
            loaded.grammar.apply(&mut loaded.scratch.logits);

            let token = sampler.sample(&mut loaded.scratch.logits, &output_tokens);

            if loaded.tokenizer.is_eos(token) {
                break;
            }

            output_tokens.push(token);

            // Early stop: semicolon followed by newline
            let piece = loaded.tokenizer.decode(&[token]);
            if piece.contains('\n') {
                let output_so_far = loaded.tokenizer.decode(&output_tokens);
                if output_so_far.contains(';') {
                    break;
                }
            }

            if pos >= ctx_size - 1 {
                break; // Context window full
            }

            // Forward pass for the new token using scratch buffers
            loaded
                .model
                .forward_into(token, pos, &mut kv_cache, &mut loaded.scratch);
            pos += 1;
        }

        let output = loaded.tokenizer.decode(&output_tokens);
        Ok(output.trim().to_string())
    }

    pub fn nl_to_sql(&self, question: &str, schema_context: &str) -> Result<String> {
        // Select only relevant tables to keep prompt short.
        // The 0.6B model degrades when given too many tables.
        let filtered_schema = filter_schema_for_question(schema_context, question);

        let user_content = if filtered_schema.is_empty() {
            question.to_string()
        } else {
            format!("{filtered_schema}\n{question}")
        };
        let prompt = format!(
            "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n\
             <|im_start|>user\n{user_content}<|im_end|>\n\
             <|im_start|>assistant\n"
        );

        // Use generate() which handles grammar constraints via soft penalty.
        // The active vocab / prefix KV cache optimizations produce identical
        // logits (verified), but generate() is simpler for now.
        let raw = self.generate(&prompt, self.max_tokens)?;

        let sql = clean_sql_output(&raw);

        if sql.is_empty() {
            return Err(TensorError::LlmError("LLM returned empty SQL".to_string()));
        }

        Ok(sql)
    }

    /// Invalidate the schema cache. Called after DDL statements.
    pub fn invalidate_schema_cache(&self) {
        self.schema_cache.invalidate();
    }

    /// Get a reference to the schema cache.
    pub fn schema_cache(&self) -> &SchemaCache {
        &self.schema_cache
    }
}

/// Filter schema context to include only tables relevant to the question.
/// The 0.6B model degrades with more than ~2 tables in context.
fn filter_schema_for_question(schema: &str, question: &str) -> String {
    if schema.is_empty() {
        return String::new();
    }

    let q_lower = question.to_lowercase();

    // Parse schema into (table_name, full_line) entries.
    // Each entry is one CREATE TABLE statement or similar block.
    let mut table_entries: Vec<(String, String)> = Vec::new();

    for line in schema.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Extract table name from various formats
        let table_name = if trimmed.to_uppercase().starts_with("CREATE TABLE ") {
            // "CREATE TABLE table_name (...)" — extract name before " ("
            let rest_original = &trimmed["CREATE TABLE ".len()..];
            rest_original
                .split(|c: char| c == '(' || c.is_whitespace())
                .next()
                .map(|s| s.to_string())
        } else if let Some(name) = trimmed.strip_prefix("Table: ") {
            Some(name.trim().to_string())
        } else if let Some(paren) = trimmed.find('(') {
            if paren > 0 && trimmed.as_bytes()[0].is_ascii_alphabetic() {
                Some(trimmed[..paren].trim().to_string())
            } else {
                None
            }
        } else {
            None
        };

        if let Some(name) = table_name {
            table_entries.push((name, trimmed.to_string()));
        }
    }

    if table_entries.is_empty() {
        return schema.to_string();
    }

    // Score each table by relevance to the question
    let mut relevant: Vec<&(String, String)> = table_entries
        .iter()
        .filter(|(name, _)| {
            let name_lower = name.to_lowercase();
            q_lower.contains(&name_lower) || q_lower.contains(name_lower.trim_end_matches('s'))
        })
        .collect();

    // If no tables matched, include all (the model needs some context)
    if relevant.is_empty() {
        relevant = table_entries.iter().collect();
    }

    // Limit to 2 tables max to keep prompt short
    relevant.truncate(2);

    relevant
        .iter()
        .map(|(_, block)| block.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

/// SQL keywords that mark the start of a statement.
const SQL_KEYWORDS: &[&str] = &[
    "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "SHOW", "DESCRIBE",
    "EXPLAIN", "WITH", "ANALYZE", "COPY",
];

fn clean_sql_output(raw: &str) -> String {
    let mut s = raw.trim().to_string();

    // Strip ChatML end-of-turn token if present
    if let Some(pos) = s.find("<|im_end|>") {
        s = s[..pos].to_string();
    }

    // Handle Qwen3 thinking blocks: <think>...</think>
    let think_content = if let Some(start) = s.find("<think>") {
        if let Some(end) = s.find("</think>") {
            let inside = s[start + 7..end].to_string();
            s = format!("{}{}", &s[..start], &s[end + 8..]);
            Some(inside)
        } else {
            let inside = s[start + 7..].to_string();
            s = s[..start].to_string();
            Some(inside)
        }
    } else {
        None
    };

    let result = extract_sql_from_text(&s);

    if result.is_empty() {
        if let Some(ref think_text) = think_content {
            return extract_sql_from_text(think_text);
        }
    }

    result
}

fn extract_sql_from_text(text: &str) -> String {
    let s = text.trim();

    let s = if s.starts_with("```sql") {
        s.strip_prefix("```sql").unwrap_or(s)
    } else if s.starts_with("```") {
        s.strip_prefix("```").unwrap_or(s)
    } else {
        s
    };
    let s = s.strip_suffix("```").unwrap_or(s).trim();

    let upper = s.to_uppercase();
    let sql_start = SQL_KEYWORDS.iter().filter_map(|kw| upper.find(kw)).min();

    let s = match sql_start {
        Some(pos) if pos > 0 => &s[pos..],
        _ => s,
    };

    if let Some(pos) = s.find(';') {
        s[..pos].trim().to_string()
    } else {
        s.trim().to_string()
    }
}

impl std::fmt::Debug for LlmEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmEngine")
            .field("model_path", &self.model_path)
            .field("loaded", &self.loaded.load(Ordering::Relaxed))
            .field("max_tokens", &self.max_tokens)
            .field("context_size", &self.context_size)
            .field("grammar_constrained", &self.grammar_constrained)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_sql_strips_fences() {
        assert_eq!(
            clean_sql_output("```sql\nSELECT * FROM users;\n```"),
            "SELECT * FROM users"
        );
    }

    #[test]
    fn clean_sql_takes_first_statement() {
        assert_eq!(clean_sql_output("SELECT 1; SELECT 2;"), "SELECT 1");
    }

    #[test]
    fn clean_sql_preserves_plain() {
        assert_eq!(clean_sql_output("SHOW TABLES;"), "SHOW TABLES");
    }

    #[test]
    fn clean_sql_handles_no_semicolon() {
        assert_eq!(
            clean_sql_output("SELECT count(*) FROM users"),
            "SELECT count(*) FROM users"
        );
    }

    #[test]
    fn clean_sql_strips_preamble_with_colon() {
        assert_eq!(
            clean_sql_output("Answer: SELECT count(*) FROM users;"),
            "SELECT count(*) FROM users"
        );
    }

    #[test]
    fn clean_sql_strips_multiline_preamble() {
        assert_eq!(
            clean_sql_output("Here is the SQL query:\n\nSELECT * FROM users WHERE balance > 500;"),
            "SELECT * FROM users WHERE balance > 500"
        );
    }

    #[test]
    fn clean_sql_strips_chatml_end_token() {
        assert_eq!(clean_sql_output("SELECT 1;<|im_end|>"), "SELECT 1");
    }

    #[test]
    fn clean_sql_strips_think_block() {
        assert_eq!(
            clean_sql_output(
                "<think>\nLet me think about this...\n</think>\nSELECT count(*) FROM users;"
            ),
            "SELECT count(*) FROM users"
        );
    }

    #[test]
    fn clean_sql_extracts_from_unclosed_think() {
        assert_eq!(
            clean_sql_output(
                "<think>\nThe SQL should be SELECT COUNT(*) FROM users WHERE balance > 500;"
            ),
            "SELECT COUNT(*) FROM users WHERE balance > 500"
        );
    }

    #[test]
    fn clean_sql_handles_show_tables() {
        assert_eq!(
            clean_sql_output("The query is: SHOW TABLES;"),
            "SHOW TABLES"
        );
    }

    #[test]
    fn clean_sql_strips_trailing_commentary() {
        assert_eq!(
            clean_sql_output("SELECT * FROM users; -- this lists all users"),
            "SELECT * FROM users"
        );
    }
}
