//! Main routing engine combining all components

use crate::circuit_breaker::CircuitBreaker;
use crate::error::{Result, TinyDancerError};
use crate::feature_engineering::FeatureEngineer;
use crate::model::FastGRNN;
use crate::types::{RouterConfig, RoutingDecision, RoutingRequest, RoutingResponse};
use crate::uncertainty::UncertaintyEstimator;
use crate::voi::{self, Belief, VoiConfig, VoiDecision};
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Instant;

/// Main router for AI agent routing
pub struct Router {
    config: RouterConfig,
    model: Arc<RwLock<FastGRNN>>,
    feature_engineer: FeatureEngineer,
    uncertainty_estimator: UncertaintyEstimator,
    circuit_breaker: Option<CircuitBreaker>,
}

impl Router {
    /// Create a new router with the given configuration
    pub fn new(config: RouterConfig) -> Result<Self> {
        // Load or create model
        let model = if std::path::Path::new(&config.model_path).exists() {
            FastGRNN::load(&config.model_path)?
        } else {
            FastGRNN::new(Default::default())?
        };

        let circuit_breaker = if config.enable_circuit_breaker {
            Some(CircuitBreaker::new(config.circuit_breaker_threshold))
        } else {
            None
        };

        // Validate the optional VoI gate at construction so route() can
        // trust it (PIR WP28, ADR-331).
        if let Some(gate) = &config.voi {
            gate.escalation.validate()?;
            VoiConfig {
                value_of_success: gate.value_of_success,
                latency_price: gate.latency_price,
            }
            .validate()?;
        }

        Ok(Self {
            config,
            model: Arc::new(RwLock::new(model)),
            feature_engineer: FeatureEngineer::new(),
            uncertainty_estimator: UncertaintyEstimator::new(),
            circuit_breaker,
        })
    }

    /// Create a router with default configuration
    pub fn default() -> Result<Self> {
        Self::new(RouterConfig::default())
    }

    /// Route a request through the system
    pub fn route(&self, request: RoutingRequest) -> Result<RoutingResponse> {
        let start = Instant::now();

        // Check circuit breaker
        if let Some(ref cb) = self.circuit_breaker {
            if !cb.is_closed() {
                return Err(TinyDancerError::CircuitBreakerError(
                    "Circuit breaker is open".to_string(),
                ));
            }
        }

        // Feature engineering
        let feature_start = Instant::now();
        let feature_vectors = self.feature_engineer.extract_batch_features(
            &request.query_embedding,
            &request.candidates,
            request.metadata.as_ref(),
        )?;
        let feature_time_us = feature_start.elapsed().as_micros() as u64;

        // Model inference
        let model = self.model.read();
        let mut decisions = Vec::new();

        for (candidate, features) in request.candidates.iter().zip(feature_vectors.iter()) {
            match model.forward(&features.features, None) {
                Ok(score) => {
                    // Estimate uncertainty
                    let uncertainty = self
                        .uncertainty_estimator
                        .estimate(&features.features, score);

                    // Determine routing decision. With a VoI gate configured,
                    // escalation to the powerful model is an estimator
                    // purchase: buy it only when the expected quality gain
                    // outweighs its cost (PIR WP28, ADR-331). Otherwise the
                    // legacy threshold rule applies unchanged.
                    let use_lightweight = match &self.config.voi {
                        Some(gate) => self.voi_use_lightweight(gate, score, uncertainty)?,
                        None => {
                            score >= self.config.confidence_threshold
                                && uncertainty <= self.config.max_uncertainty
                        }
                    };

                    decisions.push(RoutingDecision {
                        candidate_id: candidate.id.clone(),
                        confidence: score,
                        use_lightweight,
                        uncertainty,
                    });

                    // Record success with circuit breaker
                    if let Some(ref cb) = self.circuit_breaker {
                        cb.record_success();
                    }
                }
                Err(e) => {
                    // Record failure with circuit breaker
                    if let Some(ref cb) = self.circuit_breaker {
                        cb.record_failure();
                    }
                    return Err(e);
                }
            }
        }

        // Sort by confidence (descending)
        decisions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let inference_time_us = start.elapsed().as_micros() as u64;

        Ok(RoutingResponse {
            decisions,
            inference_time_us,
            candidates_processed: request.candidates.len(),
            feature_time_us,
        })
    }

    /// VoI escalation decision for one scored candidate: belief mean is the
    /// model score, belief std is the conformal uncertainty (floored so the
    /// prior stays proper), and the outside option is the configured
    /// confidence threshold. Non-finite scores are rejected here, BEFORE any
    /// comparison — a NaN score must fail loudly rather than silently pick a
    /// route.
    fn voi_use_lightweight(
        &self,
        gate: &crate::types::VoiGateConfig,
        score: f32,
        uncertainty: f32,
    ) -> Result<bool> {
        if !score.is_finite() || !uncertainty.is_finite() {
            return Err(TinyDancerError::InvalidInput(format!(
                "non-finite model output reached the VoI gate: score={score}, uncertainty={uncertainty}"
            )));
        }
        let belief = Belief::new(score as f64, f64::from(uncertainty).max(1e-6))?;
        let decision = voi::decide(
            belief,
            f64::from(self.config.confidence_threshold),
            std::slice::from_ref(&gate.escalation),
            &VoiConfig {
                value_of_success: gate.value_of_success,
                latency_price: gate.latency_price,
            },
        )?;
        Ok(decision == VoiDecision::Route)
    }

    /// Reload the model from disk
    pub fn reload_model(&self) -> Result<()> {
        let new_model = FastGRNN::load(&self.config.model_path)?;
        let mut model = self.model.write();
        *model = new_model;
        Ok(())
    }

    /// Get router configuration
    pub fn config(&self) -> &RouterConfig {
        &self.config
    }

    /// Get circuit breaker status
    pub fn circuit_breaker_status(&self) -> Option<bool> {
        self.circuit_breaker.as_ref().map(|cb| cb.is_closed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Candidate;
    use chrono::Utc;
    use std::collections::HashMap;

    #[test]
    fn test_router_creation() {
        let router = Router::default().unwrap();
        assert!(router.circuit_breaker_status().is_some());
    }

    #[test]
    fn test_routing() {
        let router = Router::default().unwrap();

        // The default FastGRNN model expects input dimension to match feature count (5)
        // Features: semantic_similarity, recency, frequency, success_rate, metadata_overlap
        let candidates = vec![
            Candidate {
                id: "1".to_string(),
                embedding: vec![0.5; 384], // Embeddings can be any size
                metadata: HashMap::new(),
                created_at: Utc::now().timestamp(),
                access_count: 10,
                success_rate: 0.95,
            },
            Candidate {
                id: "2".to_string(),
                embedding: vec![0.3; 384],
                metadata: HashMap::new(),
                created_at: Utc::now().timestamp(),
                access_count: 5,
                success_rate: 0.85,
            },
        ];

        let request = RoutingRequest {
            query_embedding: vec![0.5; 384],
            candidates,
            metadata: None,
        };

        let response = router.route(request).unwrap();
        assert_eq!(response.decisions.len(), 2);
        assert!(response.inference_time_us > 0);
    }

    #[test]
    fn test_routing_with_voi_gate() {
        use crate::types::VoiGateConfig;
        use crate::voi::EstimatorSpec;

        let request = |candidates| RoutingRequest {
            query_embedding: vec![0.5; 384],
            candidates,
            metadata: None,
        };
        let candidate = Candidate {
            id: "1".to_string(),
            embedding: vec![0.5; 384],
            metadata: HashMap::new(),
            created_at: Utc::now().timestamp(),
            access_count: 10,
            success_rate: 0.95,
        };

        // A prohibitively expensive escalation estimator is never bought:
        // every decision must fall back to the lightweight route.
        let mut config = RouterConfig::default();
        config.voi = Some(VoiGateConfig {
            value_of_success: 1.0,
            latency_price: 0.0,
            escalation: EstimatorSpec {
                cost: 1e9,
                latency_us: 0.0,
                noise_std: 0.0,
            },
        });
        let router = Router::new(config).unwrap();
        let response = router.route(request(vec![candidate.clone()])).unwrap();
        assert!(response.decisions[0].use_lightweight);

        // A free perfect estimator is always bought: escalate.
        let mut config = RouterConfig::default();
        config.voi = Some(VoiGateConfig {
            value_of_success: 1.0,
            latency_price: 0.0,
            escalation: EstimatorSpec {
                cost: 0.0,
                latency_us: 0.0,
                noise_std: 0.0,
            },
        });
        let router = Router::new(config).unwrap();
        let response = router.route(request(vec![candidate])).unwrap();
        assert!(!response.decisions[0].use_lightweight);
    }

    #[test]
    fn test_invalid_voi_gate_rejected_at_construction() {
        use crate::types::VoiGateConfig;
        use crate::voi::EstimatorSpec;

        let mut config = RouterConfig::default();
        config.voi = Some(VoiGateConfig {
            value_of_success: f64::NAN,
            latency_price: 0.0,
            escalation: EstimatorSpec {
                cost: 0.0,
                latency_us: 0.0,
                noise_std: 0.1,
            },
        });
        assert!(Router::new(config).is_err());
    }
}
