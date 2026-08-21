//! Tiny, dependency-free vector helpers shared by capability grounding and
//! causal attribution. Kept private to the crate.

/// Dot product of two equal-length slices. Extra elements on the longer slice
/// are ignored (both are truncated to the shorter length).
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm.
pub(crate) fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

/// Cosine similarity in `[-1, 1]`. Returns `0.0` when either vector is empty or
/// has zero magnitude (an undefined direction contributes no evidence).
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let na = norm(a);
    let nb = norm(b);
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot(a, b) / (na * nb)).clamp(-1.0, 1.0)
}

/// Element-wise mean of a set of equal-length vectors. Returns an empty vector
/// when the input is empty.
pub(crate) fn mean(vectors: &[Vec<f32>]) -> Vec<f32> {
    let mut iter = vectors.iter();
    let Some(first) = iter.next() else {
        return Vec::new();
    };
    let mut acc = first.clone();
    let mut count = 1.0f32;
    for v in iter {
        for (a, x) in acc.iter_mut().zip(v.iter()) {
            *a += x;
        }
        count += 1.0;
    }
    for a in acc.iter_mut() {
        *a /= count;
    }
    acc
}
