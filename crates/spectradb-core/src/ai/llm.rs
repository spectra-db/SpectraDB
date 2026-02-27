use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use parking_lot::Mutex;

use crate::error::{Result, SpectraError};

const DEFAULT_MAX_TOKENS: usize = 256;

const SYSTEM_PROMPT_TEMPLATE: &str = r#"You are a SQL translator for TensorDB (a bitemporal database).
Available tables and their schemas:
{schema_context}

TensorDB SQL supports: SELECT, INSERT, UPDATE, DELETE, CREATE TABLE,
SHOW TABLES, DESCRIBE <table>, time-travel (AS OF <timestamp>),
aggregates (count, sum, avg, min, max), JOINs, CTEs, window functions.

Translate the following question into a single SQL statement.
Output ONLY the SQL, nothing else.

Question: {question}"#;

struct LoadedModel {
    backend: LlamaBackend,
    model: LlamaModel,
}

// Safety: LlamaBackend and LlamaModel are thread-safe (the C library uses internal locking).
// We wrap them in a Mutex<Option<>> anyway, so concurrent access is serialized.
unsafe impl Send for LoadedModel {}
unsafe impl Sync for LoadedModel {}

pub struct LlmEngine {
    inner: Mutex<Option<LoadedModel>>,
    model_path: PathBuf,
    loaded: AtomicBool,
    max_tokens: usize,
}

impl LlmEngine {
    pub fn new(model_path: PathBuf) -> Self {
        Self {
            inner: Mutex::new(None),
            model_path,
            loaded: AtomicBool::new(false),
            max_tokens: DEFAULT_MAX_TOKENS,
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
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

        let backend = LlamaBackend::init()
            .map_err(|e| SpectraError::LlmError(format!("failed to init llama backend: {e}")))?;

        let model_params = LlamaModelParams::default();
        let model_params = std::pin::pin!(model_params);

        let model =
            LlamaModel::load_from_file(&backend, &self.model_path, &model_params).map_err(|e| {
                SpectraError::LlmError(format!(
                    "failed to load model {}: {e}",
                    self.model_path.display()
                ))
            })?;

        *guard = Some(LoadedModel { backend, model });
        self.loaded.store(true, Ordering::Release);
        Ok(())
    }

    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        self.ensure_loaded()?;

        let guard = self.inner.lock();
        let loaded = guard.as_ref().ok_or(SpectraError::LlmNotAvailable)?;

        let ctx_params =
            LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(2048).unwrap()));

        let mut ctx = loaded
            .model
            .new_context(&loaded.backend, ctx_params)
            .map_err(|e| SpectraError::LlmError(format!("failed to create context: {e}")))?;

        // Tokenize the prompt
        let tokens = loaded
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| SpectraError::LlmError(format!("tokenization failed: {e}")))?;

        if tokens.is_empty() {
            return Err(SpectraError::LlmError(
                "prompt tokenized to empty sequence".to_string(),
            ));
        }

        // Feed prompt tokens into context
        let mut batch = LlamaBatch::new(512, 1);
        let last_index = (tokens.len() - 1) as i32;
        for (i, token) in (0i32..).zip(tokens.iter()) {
            batch
                .add(*token, i, &[0], i == last_index)
                .map_err(|e| SpectraError::LlmError(format!("batch add failed: {e}")))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| SpectraError::LlmError(format!("initial decode failed: {e}")))?;

        // Generation loop
        let mut sampler =
            LlamaSampler::chain_simple([LlamaSampler::dist(1234), LlamaSampler::greedy()]);

        let mut output = String::new();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut n_cur = batch.n_tokens();
        let n_len = tokens.len() as i32 + max_tokens as i32;

        while n_cur <= n_len {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if loaded.model.is_eog_token(token) {
                break;
            }

            let piece = loaded
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(|e| SpectraError::LlmError(format!("token decode failed: {e}")))?;

            output.push_str(&piece);

            // Early stop: once we have a semicolon followed by a newline
            if output.contains(';') && piece.contains('\n') {
                break;
            }

            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .map_err(|e| SpectraError::LlmError(format!("batch add failed: {e}")))?;

            ctx.decode(&mut batch)
                .map_err(|e| SpectraError::LlmError(format!("decode failed: {e}")))?;

            n_cur += 1;
        }

        Ok(output.trim().to_string())
    }

    pub fn nl_to_sql(&self, question: &str, schema_context: &str) -> Result<String> {
        let prompt = SYSTEM_PROMPT_TEMPLATE
            .replace("{schema_context}", schema_context)
            .replace("{question}", question);

        let raw = self.generate(&prompt, self.max_tokens)?;

        // Clean up: strip markdown fences if present, take first statement
        let sql = clean_sql_output(&raw);

        if sql.is_empty() {
            return Err(SpectraError::LlmError("LLM returned empty SQL".to_string()));
        }

        Ok(sql)
    }
}

fn clean_sql_output(raw: &str) -> String {
    let mut s = raw.trim().to_string();

    // Strip markdown code fences
    if s.starts_with("```sql") {
        s = s.strip_prefix("```sql").unwrap_or(&s).to_string();
    } else if s.starts_with("```") {
        s = s.strip_prefix("```").unwrap_or(&s).to_string();
    }
    if s.ends_with("```") {
        s = s.strip_suffix("```").unwrap_or(&s).to_string();
    }

    let s = s.trim();

    // Take only the first SQL statement (up to and including the first semicolon)
    if let Some(pos) = s.find(';') {
        s[..=pos].trim().to_string()
    } else {
        s.to_string()
    }
}

impl std::fmt::Debug for LlmEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmEngine")
            .field("model_path", &self.model_path)
            .field("loaded", &self.loaded.load(Ordering::Relaxed))
            .field("max_tokens", &self.max_tokens)
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
            "SELECT * FROM users;"
        );
    }

    #[test]
    fn clean_sql_takes_first_statement() {
        assert_eq!(clean_sql_output("SELECT 1; SELECT 2;"), "SELECT 1;");
    }

    #[test]
    fn clean_sql_preserves_plain() {
        assert_eq!(clean_sql_output("SHOW TABLES;"), "SHOW TABLES;");
    }

    #[test]
    fn clean_sql_handles_no_semicolon() {
        assert_eq!(
            clean_sql_output("SELECT count(*) FROM users"),
            "SELECT count(*) FROM users"
        );
    }
}
