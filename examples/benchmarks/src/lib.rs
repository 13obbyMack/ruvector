//! RuVector Benchmarks Library
//!
//! Comprehensive benchmarking suite for:
//! - Temporal reasoning (TimePuzzles-style constraint inference)
//! - Vector index operations (IVF, coherence-gated search)
//! - Swarm controller regret tracking
//! - Intelligence metrics and cognitive capability assessment
//! - Adaptive learning with ReasoningBank trajectory tracking
//!
//! Based on research from:
//! - TimePuzzles benchmark (arXiv:2601.07148)
//! - Sublinear regret in multi-agent control
//! - Tool-augmented iterative temporal reasoning
//! - Cognitive capability assessment frameworks
//! - lean-agentic type theory for verified reasoning

pub mod temporal;
pub mod vector_index;
pub mod swarm_regret;
pub mod logging;
pub mod timepuzzles;
pub mod intelligence_metrics;
pub mod reasoning_bank;

pub use temporal::*;
pub use vector_index::*;
pub use swarm_regret::*;
pub use logging::*;
pub use timepuzzles::*;
pub use intelligence_metrics::*;
pub use reasoning_bank::*;
