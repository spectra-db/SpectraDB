//! x86_64 AVX-512 kernels — stubs delegating to AVX2 until Phase 3 implementation.

/// Q8_0 matvec using AVX-512 (stub — delegates to AVX2).
///
/// # Safety
/// Requires AVX-512F CPU support. Caller must ensure `data`, `input`, and
/// `output` are correctly sized for the given `rows` x `cols` dimensions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn q8_0_matvec(
    data: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    super::avx2::q8_0_matvec(data, input, output, rows, cols)
}

/// Q4_0 matvec using AVX-512 (stub — delegates to AVX2).
///
/// # Safety
/// Requires AVX-512F CPU support. Caller must ensure `data`, `input`, and
/// `output` are correctly sized for the given `rows` x `cols` dimensions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn q4_0_matvec(
    data: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    super::avx2::q4_0_matvec(data, input, output, rows, cols)
}

/// RMSNorm using AVX-512 (stub — delegates to AVX2).
///
/// # Safety
/// Requires AVX-512F CPU support. All slices must have the same length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn rms_norm(x: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    super::avx2::rms_norm(x, weight, eps, output)
}

/// SiLU activation in-place using AVX-512 (stub — delegates to AVX2).
///
/// # Safety
/// Requires AVX-512F CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn silu_inplace(x: &mut [f32]) {
    super::avx2::silu_inplace(x)
}
