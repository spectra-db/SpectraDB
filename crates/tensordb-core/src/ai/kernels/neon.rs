//! ARM64 NEON kernel implementations.
//!
//! All functions require NEON support, enforced via `#[target_feature]`.
//! The caller must ensure the CPU supports these features before invoking
//! (handled by the dispatch layer in `mod.rs`).

// SIMD kernels use index-based loops for clarity with offset calculations.
#![allow(clippy::needless_range_loop)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Q8_0 matrix-vector multiply ─────────────────────────────────────────

/// NEON Q8_0 matvec -- processes 32 elements per block in 8 groups of 4.
///
/// Block layout: 2-byte f16 scale + 32 int8 quants = 34 bytes per block.
///
/// # Safety
/// Requires NEON CPU support. Caller must ensure `data`, `input`, and
/// `output` are correctly sized for the given `rows` x `cols` dimensions.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn q8_0_matvec(
    data: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34;
    let blocks_per_row = cols / BLOCK_SIZE;

    for r in 0..rows {
        let mut acc = vdupq_n_f32(0.0);
        let row_offset = r * blocks_per_row * BLOCK_BYTES;

        for b in 0..blocks_per_row {
            let block = &data[row_offset + b * BLOCK_BYTES..];
            let scale =
                half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();

            let input_offset = b * BLOCK_SIZE;
            let quants = &block[2..];

            let mut block_acc = vdupq_n_f32(0.0);

            // Process 32 elements in 8 groups of 4
            for g in 0..8 {
                let g_off = g * 4;

                // Load 4 int8 values via an 8-byte NEON load (we use the low 4)
                let q_ptr = quants[g_off..].as_ptr() as *const i8;
                let q_i8 = vld1_s8(q_ptr);
                // Sign-extend i8 -> i16
                let q_i16 = vmovl_s8(q_i8);
                // Sign-extend low 4 x i16 -> 4 x i32
                let q_i32 = vmovl_s16(vget_low_s16(q_i16));
                // Convert i32 -> f32
                let q_f32 = vcvtq_f32_s32(q_i32);

                // Load 4 input f32 values
                let inp = vld1q_f32(input[input_offset + g_off..].as_ptr());

                // FMA: block_acc += q * inp
                block_acc = vfmaq_f32(block_acc, q_f32, inp);
            }

            let scale_v = vdupq_n_f32(scale);
            acc = vfmaq_f32(acc, scale_v, block_acc);
        }

        // Horizontal sum of 4-wide accumulator
        output[r] = vaddvq_f32(acc);
    }
}

// ── Q4_0 matrix-vector multiply ─────────────────────────────────────────

/// NEON Q4_0 matvec -- nibble unpacking with FMA accumulation.
///
/// Block layout: 2-byte f16 scale + 16 packed bytes (32 nibbles) = 18 bytes.
/// Each byte packs two 4-bit values: lo nibble first, hi nibble second.
/// Values are unsigned [0,15] biased by -8 to give signed range [-8, 7].
///
/// # Safety
/// Requires NEON CPU support. Caller must ensure slices are correctly sized.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn q4_0_matvec(
    data: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18;
    let blocks_per_row = cols / BLOCK_SIZE;

    let lo_mask_v = vdupq_n_u8(0x0F);
    let bias_v = vdupq_n_s16(8);

    for r in 0..rows {
        let mut acc = vdupq_n_f32(0.0);
        let row_offset = r * blocks_per_row * BLOCK_BYTES;

        for b in 0..blocks_per_row {
            let block = &data[row_offset + b * BLOCK_BYTES..];
            let scale =
                half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();

            let input_offset = b * BLOCK_SIZE;
            let quants = &block[2..];

            let mut block_acc = vdupq_n_f32(0.0);

            // Process 16 bytes = 32 nibbles in 4 groups of 4 bytes (8 nibbles each).
            // Each group of 4 bytes yields 8 values paired with 8 input floats,
            // accumulated as 4 pair-sums.
            for g in 0..4 {
                let g_off = g * 4;
                let base = input_offset + g * 8;
                let q_ptr = quants[g_off..].as_ptr();

                // Load 4 packed bytes (we load 8 via vld1_u8, use low 4)
                let raw = vld1_u8(q_ptr);

                // Extract lo nibbles: byte & 0x0F (using the low 8 bytes of mask)
                let lo_u8 = vand_u8(raw, vget_low_u8(lo_mask_v));
                // Extract hi nibbles: byte >> 4
                let hi_u8 = vand_u8(vshr_n_u8(raw, 4), vget_low_u8(lo_mask_v));

                // Zero-extend first 4 u8 -> i16
                let lo_u16 = vmovl_u8(lo_u8);
                let lo_i16 = vreinterpretq_s16_u16(lo_u16);
                // Take low 4 values and subtract bias
                let lo_biased = vsub_s16(vget_low_s16(lo_i16), vget_low_s16(bias_v));
                // Extend to i32 and convert to f32
                let lo_i32 = vmovl_s16(lo_biased);
                let lo_f32 = vcvtq_f32_s32(lo_i32);

                let hi_u16 = vmovl_u8(hi_u8);
                let hi_i16 = vreinterpretq_s16_u16(hi_u16);
                let hi_biased = vsub_s16(vget_low_s16(hi_i16), vget_low_s16(bias_v));
                let hi_i32 = vmovl_s16(hi_biased);
                let hi_f32 = vcvtq_f32_s32(hi_i32);

                // Gather even/odd inputs for the 4 pairs
                let inp_ptr = input[base..].as_ptr();
                let mut even = [0.0f32; 4];
                let mut odd = [0.0f32; 4];
                for j in 0..4 {
                    even[j] = *inp_ptr.add(j * 2);
                    odd[j] = *inp_ptr.add(j * 2 + 1);
                }
                let even_v = vld1q_f32(even.as_ptr());
                let odd_v = vld1q_f32(odd.as_ptr());

                // pair_sum = lo * even + hi * odd
                let pair_sum = vfmaq_f32(vmulq_f32(hi_f32, odd_v), lo_f32, even_v);
                block_acc = vaddq_f32(block_acc, pair_sum);
            }

            let scale_v = vdupq_n_f32(scale);
            acc = vfmaq_f32(acc, scale_v, block_acc);
        }

        // Horizontal sum
        output[r] = vaddvq_f32(acc);
    }
}

// ── RMSNorm ─────────────────────────────────────────────────────────────

/// NEON RMSNorm: `output[i] = weight[i] * (x[i] / rms)`.
///
/// Computes the root-mean-square using vectorized sum-of-squares, then applies
/// the normalization and weight scaling.
///
/// # Safety
/// Requires NEON CPU support. All slices must have the same length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn rms_norm(x: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let n = x.len();

    // ---- Vectorized sum-of-squares ----
    let mut ss_acc = vdupq_n_f32(0.0);
    let chunks = n / 4;

    for i in 0..chunks {
        let v = vld1q_f32(x[i * 4..].as_ptr());
        ss_acc = vfmaq_f32(ss_acc, v, v);
    }

    // Horizontal sum
    let mut ss = vaddvq_f32(ss_acc);

    // Handle remainder elements
    for i in (chunks * 4)..n {
        ss += x[i] * x[i];
    }

    let inv_rms = 1.0 / (ss / n as f32 + eps).sqrt();
    let inv_rms_v = vdupq_n_f32(inv_rms);

    // ---- Apply normalization + weight ----
    for i in 0..chunks {
        let off = i * 4;
        let xv = vld1q_f32(x[off..].as_ptr());
        let wv = vld1q_f32(weight[off..].as_ptr());
        let normed = vmulq_f32(xv, inv_rms_v);
        let result = vmulq_f32(wv, normed);
        vst1q_f32(output[off..].as_mut_ptr(), result);
    }

    // Handle remainder elements
    for i in (chunks * 4)..n {
        output[i] = weight[i] * (x[i] * inv_rms);
    }
}

// ── SiLU ────────────────────────────────────────────────────────────────

/// SiLU (Sigmoid Linear Unit) activation, applied in-place (scalar fallback).
///
/// `x[i] = x[i] / (1.0 + exp(-x[i]))` which is equivalent to `x * sigmoid(x)`.
/// The exp() function is not directly available as a NEON intrinsic, so this
/// uses the scalar implementation.
///
/// # Safety
/// Requires NEON CPU support (for target_feature consistency), but the
/// actual computation is scalar.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}
