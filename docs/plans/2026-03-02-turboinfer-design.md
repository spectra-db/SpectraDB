# TurboInfer: Blazing-Fast NL2SQL Inference Engine

**Date**: 2026-03-02
**Status**: Approved
**Goal**: Make TensorDB's embedded inference engine faster than llama.cpp for the NL2SQL use case

## Problem

The prem-1B-SQL model (1.3B params, hidden_dim=2048, 24 layers) is unusable on CPU with scalar matvec — a single query takes 30+ seconds. Even with rayon parallelism, the per-token cost (~800ms) makes interactive use painful.

## Strategy: Three Layers of Optimization

### Layer 1: World-Class SIMD Kernels (Match llama.cpp)

**Remove the `simd` feature flag.** All SIMD code compiles unconditionally. At startup, probe CPU features once and build a dispatch table.

#### Runtime CPU Feature Detection

```rust
struct CpuFeatures {
    // ARM64
    has_neon: bool,       // 128-bit SIMD (baseline)
    has_i8mm: bool,       // INT8 matrix multiply (smmla)
    has_sve2: bool,       // Scalable Vector Extension 2
    has_dotprod: bool,    // SDOT instruction
    // x86_64
    has_avx2: bool,       // 256-bit FMA
    has_avx512f: bool,    // 512-bit float
    has_avx512vnni: bool, // INT8 dot product (vpdpbusd)
    // Common
    l1_cache_size: usize,
    num_cores: usize,
}
```

Detected once at startup via `std::arch::is_aarch64_feature_detected!` / `is_x86_feature_detected!`. Stored in a global `OnceLock<CpuFeatures>`.

#### ARM64 Kernel Tiers

| Kernel | Instruction | Elements/cycle | Speedup vs scalar |
|--------|-------------|---------------|-------------------|
| Q8_0 I8MM | `smmla` | 8 int8 MAC/instr | 8-10x |
| Q8_0 SDOT | `vsdotq_s32` | 4 int8 MAC/instr | 4-5x |
| Q8_0 NEON | `vmla` | 4 f32 FMA/instr | 3-4x (existing) |
| Q4_0 I8MM | `smmla` + nibble unpack | 8 int8 MAC/instr | 6-8x |
| Q4_0 NEON | manual nibble + `vmla` | 4 f32 FMA/instr | 2-3x |
| RMSNorm NEON | `vrsqrteq_f32` + `vfmaq` | 4 f32/instr | 3x |
| SiLU NEON | approximated exp | 4 f32/instr | 2-3x |

**I8MM is the crown jewel.** The `smmla` instruction takes two 8x1 int8 vectors, multiplies elementwise, and accumulates into 32-bit. This is exactly what Q8_0 matvec needs — no f16→f32 scale conversion per element, just raw int8 multiply-accumulate, then apply scale once per block.

#### x86_64 Kernel Tiers

| Kernel | Instruction | Elements/cycle | Speedup vs scalar |
|--------|-------------|---------------|-------------------|
| Q8_0 VNNI | `_mm256_dpbusd_epi32` | 8 int8 MAC/instr | 6-8x |
| Q8_0 AVX-512 | `_mm512_fmadd_ps` | 16 f32 FMA/instr | 5-6x |
| Q8_0 AVX2 | `_mm256_fmadd_ps` | 8 f32 FMA/instr | 3-4x (existing) |
| Q4_0 VNNI | `_mm256_dpbusd_epi32` + nibble | 8 int8 MAC/instr | 5-7x |
| Q4_0 AVX-512 | `_mm512_fmadd_ps` | 16 f32 FMA/instr | 4-5x |

#### Cache-Line Tiled Execution

Partition matvec so each tile's input slice fits in L1 cache. Process all rows for a column tile before advancing to the next column tile:

```
For each col_tile [0..cols, step=TILE]:
  Load input[tile..tile+TILE] into L1
  For each row r (rayon-parallel):
    partial_sum[r] += dot(W[r][tile..tile+TILE], input[tile..tile+TILE])
```

TILE size = L1_size / sizeof(f32) / 2 ≈ 4096 elements. Input chunk stays hot in L1 across all rows. Weight data streams sequentially (prefetch-friendly).

### Layer 2: NL2SQL Architectural Wins (Beat llama.cpp)

These are optimizations that only work because we know we're generating SQL, not arbitrary text. llama.cpp can't do these structurally.

| Optimization | Mechanism | Speedup |
|-------------|-----------|---------|
| Permanent system prompt KV cache | System prompt (~50 tokens) computed once per model load, persisted in memory. Never recomputed. | Saves 50 forward passes on first query |
| Active vocab pruning | Already implemented — 5K of 151K tokens | 30x on LM head (existing) |
| Schema KV cache reuse | Already implemented — reuses prefix | ~50% on repeat queries (existing) |
| Fused RMSNorm→Matvec | Compute `x[i] * inv_rms * weight[i]` during matvec input read, not as separate pass | 1.3-1.5x per layer |
| Batched QKV projection | Q, K, V share same input. Concatenate weight matrices, single matvec, split output. | ~1.3x (one input read not three) |
| Parallel attention heads | GQA heads are independent — rayon across heads for long sequences | 2-4x for attention at seq_len > 128 |
| Early semicolon termination | Already implemented | Saves ~10-20 tokens |

#### Fused RMSNorm→Matvec

Instead of:
1. RMSNorm: read x → compute rms → write xb (full memory round-trip)
2. Matvec: read xb → compute dot products → write output

Fuse into:
1. Compute rms (one pass over x) → get inv_rms scalar
2. Matvec: read x, multiply by inv_rms * norm_weight on the fly during dot product

Eliminates one full hidden_dim read+write cycle per layer (2 × 2048 × 4 = 16KB saved per layer, 24 layers = 384KB of memory bandwidth saved per token).

#### Batched QKV Projection

Currently: 3 separate matvec calls (Q, K, V), each reading the same input.
Proposed: Concatenate Q/K/V weight rows into one matrix, single matvec, split output.

This means the input vector is read from cache once instead of three times. For hidden_dim=2048:
- Current: 3 × 2048 cache line reads = 6144 reads
- Batched: 1 × 2048 cache line reads = 2048 reads (but with 3x more rows)
- Net benefit: ~30% from better input cache utilization

### Layer 3: SQL Template Speculation (Novel)

Zero-cost speculative decoding using SQL grammar knowledge instead of a draft model.

#### How It Works

1. **Pre-tokenize ~40 SQL templates** filled with schema-specific table/column names:
   ```
   "SELECT COUNT(*) FROM users;"
   "SELECT * FROM orders WHERE status = '"
   "SELECT u.name, o.amount FROM users u JOIN orders o ON u.id = o.user_id"
   ```

2. **Match current generation prefix** against templates to find likely continuations.

3. **Draft 2-8 tokens** (adaptive window based on SQL parse state confidence).

4. **Verify via batched forward pass** — process all draft tokens in one pass (same cost as generating one token).

5. **Accept consecutive correct tokens**, reject at first divergence. Use model's choice at divergence point.

#### Correctness Guarantee

Speculative decoding is mathematically equivalent to sequential generation — verification uses the same model with the same logits. The draft just *proposes* candidates; the model has final say. No quality degradation.

#### Adaptive Speculation Window

- After `FROM`, `WHERE`, `JOIN`, `ON` → 8 tokens (highly constrained)
- After `SELECT`, `,`, operators → 4 tokens (moderately constrained)
- After values, string literals → 1-2 tokens (unpredictable)

#### Expected Acceptance Rate

For SQL: 60-80% of speculated tokens accepted. Each accepted token is "free" (no additional forward pass). Effective generation speedup: 2.5-4x.

## Projected Performance

For prem-1B-SQL Q8_0, generating a 15-token SQL query:

| Phase | Current (scalar) | After optimization | Speedup |
|-------|------------------|--------------------|---------|
| System prompt prefill (50 tok) | ~50 × 800ms = 40s | 0 (cached) | ∞ |
| Schema prefill (30 tok) | ~30 × 800ms = 24s | 0 (KV reuse) | ∞ |
| Question prefill (20 tok) | 20 × 800ms = 16s | 20 × 100ms = 2s | 8x |
| Generation (15 tok) | 15 × 800ms = 12s | 5 × 100ms = 0.5s | 24x |
| **Total first query** | **~30s** | **~2.5s** | **12x** |
| **Total repeat query** | **~28s** | **~0.5s** | **56x** |

## Files to Create/Modify

| File | Change |
|------|--------|
| `crates/tensordb-core/src/ai/kernels.rs` | NEW: Kernel dispatch engine, CPU feature detection |
| `crates/tensordb-core/src/ai/kernels/` | NEW: Directory with per-arch kernel files |
| `crates/tensordb-core/src/ai/kernels/neon.rs` | NEW: ARM64 NEON kernels (Q8_0, Q4_0, RMSNorm, SiLU) |
| `crates/tensordb-core/src/ai/kernels/i8mm.rs` | NEW: ARM64 I8MM kernels (Q8_0, Q4_0) |
| `crates/tensordb-core/src/ai/kernels/avx2.rs` | MOVE: Existing AVX2 kernels from transformer.rs |
| `crates/tensordb-core/src/ai/kernels/avx512.rs` | NEW: AVX-512 + VNNI kernels |
| `crates/tensordb-core/src/ai/kernels/scalar.rs` | MOVE: Existing scalar kernels from transformer.rs |
| `crates/tensordb-core/src/ai/transformer.rs` | MODIFY: Use kernel dispatch instead of inline SIMD, add fused ops |
| `crates/tensordb-core/src/ai/speculation.rs` | NEW: SQL template speculative decoding |
| `crates/tensordb-core/src/ai/llm.rs` | MODIFY: Integrate speculation into generation loop |
| `crates/tensordb-core/Cargo.toml` | MODIFY: Remove `simd` feature flag |
| `benches/inference.rs` | MODIFY: Add kernel comparison benchmarks |

## Build Sequence

1. **Phase 1**: Kernel engine + CPU detection + move existing SIMD code
2. **Phase 2**: ARM64 I8MM kernels + cache tiling
3. **Phase 3**: x86_64 AVX-512/VNNI kernels
4. **Phase 4**: Fused RMSNorm→Matvec + Batched QKV
5. **Phase 5**: SQL template speculation engine
6. **Phase 6**: Integration + benchmarking + tuning

## Success Criteria

- prem-1B-SQL generates correct SQL for "How many users?" in under 3 seconds (first query)
- Repeat queries with same schema complete in under 1 second
- Full 21-question scorecard completes in under 60 seconds
- All existing tests pass (correctness preserved)
- No regression on Qwen3-0.6B performance
