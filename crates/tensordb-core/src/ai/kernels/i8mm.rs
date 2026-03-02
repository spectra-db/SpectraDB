//! ARM64 I8MM kernels — stubs delegating to NEON until Phase 2 implementation.

/// Q8_0 matvec using I8MM (stub — delegates to NEON).
///
/// # Safety
/// Requires NEON CPU support. Caller must ensure `data`, `input`, and
/// `output` are correctly sized for the given `rows` x `cols` dimensions.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn q8_0_matvec(data: &[u8], input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    super::neon::q8_0_matvec(data, input, output, rows, cols)
}

/// Q4_0 matvec using I8MM (stub — delegates to NEON).
///
/// # Safety
/// Requires NEON CPU support. Caller must ensure `data`, `input`, and
/// `output` are correctly sized for the given `rows` x `cols` dimensions.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn q4_0_matvec(data: &[u8], input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    super::neon::q4_0_matvec(data, input, output, rows, cols)
}
