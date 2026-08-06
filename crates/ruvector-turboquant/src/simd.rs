//! Integer dot-product kernels over packed Turbo4 codes.
//!
//! Two primitives, each with a scalar oracle and an AVX2 path selected at
//! runtime (NEON / AVX-512 VNNI / WASM SIMD128 are phase-3 of ADR-296 and
//! plug into the same dispatch seam):
//!
//! * [`dot_i8_nibble`] — asymmetric: int8 query (`D` lanes, contiguous)
//!   × packed nibble code (`D/2` bytes, low nibbles = dims `0..D/2`,
//!   high nibbles = dims `D/2..D`).
//! * [`dot_nibble_nibble`] — symmetric: packed code × packed code.
//!
//! Nibbles index [`LEVELS_I8`], the 16-entry int8 Lloyd-Max grid that fits a
//! single `pshufb` register. Products are i8×i8 (≤ 16 129), accumulated
//! exactly in i32 — the kernels are bit-exact vs the scalar oracle, which the
//! tests enforce.

use crate::tables::LEVELS_I8;

/// Asymmetric dot: Σᵢ q[i] · L_i8[code[i]] with q as raw-i8 bytes.
/// `nibbles.len() == dim/2`, `q.len() == dim`.
#[inline]
pub fn dot_i8_nibble(nibbles: &[u8], q: &[u8], dim: usize) -> i32 {
    debug_assert_eq!(nibbles.len(), dim / 2);
    debug_assert!(q.len() >= dim);
    #[cfg(target_arch = "x86_64")]
    {
        if dim >= 64 && is_x86_feature_detected!("avx2") {
            return unsafe { dot_i8_nibble_avx2(nibbles, q, dim) };
        }
    }
    dot_i8_nibble_scalar(nibbles, q, dim)
}

/// Symmetric dot: Σᵢ L_i8[a[i]] · L_i8[b[i]] over two packed codes.
#[inline]
pub fn dot_nibble_nibble(a: &[u8], b: &[u8], dim: usize) -> i32 {
    debug_assert_eq!(a.len(), dim / 2);
    debug_assert_eq!(b.len(), dim / 2);
    #[cfg(target_arch = "x86_64")]
    {
        if dim >= 64 && is_x86_feature_detected!("avx2") {
            return unsafe { dot_nibble_nibble_avx2(a, b, dim) };
        }
    }
    dot_nibble_nibble_scalar(a, b, dim)
}

pub(crate) fn dot_i8_nibble_scalar(nibbles: &[u8], q: &[u8], dim: usize) -> i32 {
    let half = dim / 2;
    let mut acc = 0i32;
    for i in 0..half {
        let lo = LEVELS_I8[(nibbles[i] & 0x0F) as usize] as i32;
        let hi = LEVELS_I8[(nibbles[i] >> 4) as usize] as i32;
        acc += (q[i] as i8 as i32) * lo + (q[i + half] as i8 as i32) * hi;
    }
    acc
}

pub(crate) fn dot_nibble_nibble_scalar(a: &[u8], b: &[u8], dim: usize) -> i32 {
    let half = dim / 2;
    let mut acc = 0i32;
    for i in 0..half {
        let al = LEVELS_I8[(a[i] & 0x0F) as usize] as i32;
        let ah = LEVELS_I8[(a[i] >> 4) as usize] as i32;
        let bl = LEVELS_I8[(b[i] & 0x0F) as usize] as i32;
        let bh = LEVELS_I8[(b[i] >> 4) as usize] as i32;
        acc += al * bl + ah * bh;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::LEVELS_I8;
    use std::arch::x86_64::*;

    /// The 16-entry level grid broadcast to both 128-bit lanes for
    /// `_mm256_shuffle_epi8` (which shuffles within lanes).
    #[inline]
    unsafe fn level_table() -> __m256i {
        let t = _mm_loadu_si128(LEVELS_I8.as_ptr() as *const __m128i);
        _mm256_broadcastsi128_si256(t)
    }

    /// Multiply signed i8 lanes of `a` and `b`, accumulating i32 into `acc`.
    #[inline]
    unsafe fn madd_i8(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
        // Widen each 128-bit half to i16 and use madd_epi16 (i16×i16 → i32 pairs).
        let a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a));
        let a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1));
        let b_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b));
        let b_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 1));
        let p = _mm256_add_epi32(
            _mm256_madd_epi16(a_lo, b_lo),
            _mm256_madd_epi16(a_hi, b_hi),
        );
        _mm256_add_epi32(acc, p)
    }

    #[inline]
    unsafe fn hsum_i32(v: __m256i) -> i32 {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256(v, 1);
        let s = _mm_add_epi32(lo, hi);
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01));
        _mm_cvtsi128_si32(s)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_i8_nibble_avx2(nibbles: &[u8], q: &[u8], dim: usize) -> i32 {
        let half = dim / 2;
        let table = level_table();
        let mask = _mm256_set1_epi8(0x0F);
        let mut acc = _mm256_setzero_si256();

        let chunks = half / 32;
        for c in 0..chunks {
            let i = c * 32;
            let packed = _mm256_loadu_si256(nibbles.as_ptr().add(i) as *const __m256i);
            // Low nibbles → levels for dims i..i+32 (contiguous run 1).
            let lo_idx = _mm256_and_si256(packed, mask);
            let lo_lev = _mm256_shuffle_epi8(table, lo_idx);
            let q_lo = _mm256_loadu_si256(q.as_ptr().add(i) as *const __m256i);
            acc = madd_i8(acc, lo_lev, q_lo);
            // High nibbles → levels for dims half+i..half+i+32 (run 2).
            let hi_idx = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
            let hi_lev = _mm256_shuffle_epi8(table, hi_idx);
            let q_hi = _mm256_loadu_si256(q.as_ptr().add(half + i) as *const __m256i);
            acc = madd_i8(acc, hi_lev, q_hi);
        }

        let mut total = hsum_i32(acc);
        // Scalar tail over remaining packed bytes.
        for i in chunks * 32..half {
            let lo = LEVELS_I8[(nibbles[i] & 0x0F) as usize] as i32;
            let hi = LEVELS_I8[(nibbles[i] >> 4) as usize] as i32;
            total += (q[i] as i8 as i32) * lo + (q[i + half] as i8 as i32) * hi;
        }
        total
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_nibble_nibble_avx2(a: &[u8], b: &[u8], dim: usize) -> i32 {
        let half = dim / 2;
        let table = level_table();
        let mask = _mm256_set1_epi8(0x0F);
        let mut acc = _mm256_setzero_si256();

        let chunks = half / 32;
        for c in 0..chunks {
            let i = c * 32;
            let pa = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let pb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let a_lo = _mm256_shuffle_epi8(table, _mm256_and_si256(pa, mask));
            let b_lo = _mm256_shuffle_epi8(table, _mm256_and_si256(pb, mask));
            acc = madd_i8(acc, a_lo, b_lo);
            let a_hi = _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16(pa, 4), mask));
            let b_hi = _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16(pb, 4), mask));
            acc = madd_i8(acc, a_hi, b_hi);
        }

        let mut total = hsum_i32(acc);
        for i in chunks * 32..half {
            let al = LEVELS_I8[(a[i] & 0x0F) as usize] as i32;
            let ah = LEVELS_I8[(a[i] >> 4) as usize] as i32;
            let bl = LEVELS_I8[(b[i] & 0x0F) as usize] as i32;
            let bh = LEVELS_I8[(b[i] >> 4) as usize] as i32;
            total += al * bl + ah * bh;
        }
        total
    }
}

#[cfg(target_arch = "x86_64")]
use avx2::{dot_i8_nibble_avx2, dot_nibble_nibble_avx2};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rotation::SplitMix64;

    fn random_code(half: usize, seed: u64) -> Vec<u8> {
        let mut rng = SplitMix64(seed);
        (0..half).map(|_| (rng.next_u64() & 0xFF) as u8).collect()
    }

    fn random_q(dim: usize, seed: u64) -> Vec<u8> {
        let mut rng = SplitMix64(seed);
        (0..dim).map(|_| (rng.next_u64() & 0xFF) as u8).collect()
    }

    #[test]
    fn simd_matches_scalar_oracle() {
        // Cover multiple sizes incl. non-multiples of 64 dims (tail path).
        for dim in [64usize, 128, 192, 384, 1536, 100, 70] {
            let dim = dim & !1; // even
            let half = dim / 2;
            for seed in 0..5u64 {
                let code_a = random_code(half, seed * 3 + 1);
                let code_b = random_code(half, seed * 3 + 2);
                let q = random_q(dim, seed * 3 + 3);
                assert_eq!(
                    dot_i8_nibble(&code_a, &q, dim),
                    dot_i8_nibble_scalar(&code_a, &q, dim),
                    "asym dim {dim} seed {seed}"
                );
                assert_eq!(
                    dot_nibble_nibble(&code_a, &code_b, dim),
                    dot_nibble_nibble_scalar(&code_a, &code_b, dim),
                    "sym dim {dim} seed {seed}"
                );
            }
        }
    }

    #[test]
    fn known_small_case() {
        // dim 4: dims 0,1 in low nibbles of bytes 0,1; dims 2,3 in high.
        // code: dim0=15 (level 127), dim1=0 (level -127), dim2=8 (6), dim3=7 (-6)
        let code = vec![0x8F, 0x70];
        let q = vec![1i8 as u8, 2i8 as u8, 3i8 as u8, (-4i8) as u8];
        let expect = 1 * 127 + 2 * -127 + 3 * 6 + -4 * -6;
        assert_eq!(dot_i8_nibble_scalar(&code, &q, 4), expect);
    }
}
