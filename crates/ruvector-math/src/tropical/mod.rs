//! Tropical Algebra (Max-Plus Semiring)
//!
//! Tropical algebra replaces (×, +) with (max, +) or (min, +).
//! Applications:
//! - Neural network analysis (piecewise linear functions)
//! - Shortest path algorithms
//! - Dynamic programming
//! - Linear programming duality
//!
//! ## Mathematical Background
//!
//! The tropical semiring (ℝ ∪ {-∞}, ⊕, ⊗) where:
//! - a ⊕ b = max(a, b)
//! - a ⊗ b = a + b
//! - Zero element: -∞
//! - Unit element: 0
//!
//! ## Key Results
//!
//! - Tropical polynomials are piecewise linear
//! - Neural networks with ReLU = tropical rational functions
//! - Tropical geometry provides bounds on linear regions

mod semiring;
mod polynomial;
mod matrix;
mod neural_analysis;

pub use semiring::{Tropical, TropicalSemiring};
pub use polynomial::{TropicalPolynomial, TropicalMonomial};
pub use matrix::{TropicalMatrix, TropicalEigen, MinPlusMatrix};
pub use neural_analysis::{LinearRegionCounter, TropicalNeuralAnalysis};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tropical_ops() {
        let a = Tropical::new(3.0);
        let b = Tropical::new(5.0);

        assert_eq!(a.add(&b).value(), 5.0); // max(3, 5) = 5
        assert_eq!(a.mul(&b).value(), 8.0); // 3 + 5 = 8
    }
}
