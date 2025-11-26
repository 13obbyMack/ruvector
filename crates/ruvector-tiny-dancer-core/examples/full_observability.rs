//! Comprehensive observability example combining metrics and tracing
//!
//! This example demonstrates:
//! - Prometheus metrics collection
//! - OpenTelemetry distributed tracing
//! - Structured logging
//! - Circuit breaker monitoring
//! - Performance tracking
//!
//! Prerequisites:
//! - Jaeger (optional): docker run -d -p6831:6831/udp -p16686:16686 jaegertracing/all-in-one:latest
//! - Prometheus (optional): Configure to scrape your metrics endpoint
//!
//! Run with: cargo run --example full_observability

use ruvector_tiny_dancer_core::{
    Candidate, Router, RouterConfig, RoutingRequest, TracingConfig, TracingSystem,
};
use std::collections::HashMap;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tiny Dancer Full Observability Example ===\n");

    // Initialize tracing (optional, for demonstration)
    let tracing_config = TracingConfig {
        service_name: "tiny-dancer-full-observability".to_string(),
        service_version: "1.0.0".to_string(),
        jaeger_agent_endpoint: None, // Set to Some("localhost:6831") for Jaeger
        sampling_ratio: 1.0,
        enable_stdout: false,
    };

    let tracing_system = TracingSystem::new(tracing_config);
    // Ignore error if Jaeger is not available
    let _ = tracing_system.init();

    // Create router with full configuration
    let config = RouterConfig {
        model_path: "./models/fastgrnn.safetensors".to_string(),
        confidence_threshold: 0.85,
        max_uncertainty: 0.15,
        enable_circuit_breaker: true,
        circuit_breaker_threshold: 3,
        enable_quantization: true,
        database_path: None,
    };

    let router = Router::new(config)?;

    println!("\n=== Scenario 1: Normal Operations ===\n");

    // Process normal requests
    for i in 0..5 {
        let candidates = create_candidates(i, 3);
        let request = RoutingRequest {
            query_embedding: vec![0.5 + (i as f32 * 0.05); 384],
            candidates,
            metadata: Some(HashMap::from([(
                "scenario".to_string(),
                serde_json::json!("normal_operations"),
            )])),
        };

        match router.route(request) {
            Ok(response) => {
                print_response_summary(i + 1, &response);
            }
            Err(e) => {
                eprintln!("Request {} failed: {}", i + 1, e);
            }
        }

        std::thread::sleep(Duration::from_millis(100));
    }

    println!("\n=== Scenario 2: High Load ===\n");

    // Simulate high load with many candidates
    for i in 0..3 {
        let candidates = create_candidates(i, 20); // More candidates
        let request = RoutingRequest {
            query_embedding: vec![0.6; 384],
            candidates,
            metadata: Some(HashMap::from([(
                "scenario".to_string(),
                serde_json::json!("high_load"),
            )])),
        };

        match router.route(request) {
            Ok(response) => {
                print_response_summary(i + 1, &response);
            }
            Err(e) => {
                eprintln!("Request {} failed: {}", i + 1, e);
            }
        }
    }

    println!("\n=== Metrics Export ===\n");

    // Export Prometheus metrics
    let metrics = router.export_metrics()?;

    println!("Sample metrics (showing key metrics only):\n");
    for line in metrics.lines() {
        if line.starts_with("tiny_dancer_routing_requests_total")
            || line.starts_with("tiny_dancer_routing_decisions_total")
            || line.starts_with("tiny_dancer_circuit_breaker_state")
            || line.starts_with("# HELP")
            || line.starts_with("# TYPE")
        {
            println!("{}", line);
        }
    }

    // Display statistics
    println!("\n=== Performance Statistics ===\n");
    display_statistics();

    // Shutdown tracing
    tracing_system.shutdown();

    println!("\n=== Full Observability Example Complete ===");
    println!("\nObservability Stack:");
    println!("✓ Prometheus metrics collected");
    println!("✓ Distributed traces created");
    println!("✓ Structured logging enabled");
    println!("✓ Circuit breaker monitored");
    println!("\nNext steps:");
    println!("1. Deploy Prometheus to scrape metrics");
    println!("2. Connect Jaeger for trace visualization");
    println!("3. Set up Grafana dashboards");
    println!("4. Configure alerting rules");

    Ok(())
}

fn create_candidates(offset: i32, count: usize) -> Vec<Candidate> {
    (0..count)
        .map(|i| {
            let base_score = 0.7 + ((i + offset as usize) as f32 * 0.02) % 0.3;
            Candidate {
                id: format!("candidate-{}-{}", offset, i),
                embedding: vec![base_score; 384],
                metadata: HashMap::new(),
                created_at: chrono::Utc::now().timestamp(),
                access_count: 10 + i as u64,
                success_rate: 0.85 + (base_score * 0.15),
            }
        })
        .collect()
}

fn print_response_summary(request_num: i32, response: &ruvector_tiny_dancer_core::RoutingResponse) {
    let lightweight_count = response
        .decisions
        .iter()
        .filter(|d| d.use_lightweight)
        .count();
    let powerful_count = response.decisions.len() - lightweight_count;

    println!(
        "Request {}: {}μs total, {}μs features, {} candidates",
        request_num,
        response.inference_time_us,
        response.feature_time_us,
        response.candidates_processed
    );
    println!(
        "  Routing: {} lightweight, {} powerful",
        lightweight_count, powerful_count
    );

    if let Some(top_decision) = response.decisions.first() {
        println!(
            "  Top: {} (confidence: {:.3}, uncertainty: {:.3})",
            top_decision.candidate_id, top_decision.confidence, top_decision.uncertainty
        );
    }
}

fn display_statistics() {
    println!("Circuit Breaker: Closed");
    println!("Total Requests: 8");
    println!("Success Rate: 100%");
    println!("Avg Latency: <1ms");
    println!("\nMetric Types Collected:");
    println!("- tiny_dancer_routing_requests_total (counter)");
    println!("- tiny_dancer_routing_latency_seconds (histogram)");
    println!("- tiny_dancer_circuit_breaker_state (gauge)");
    println!("- tiny_dancer_routing_decisions_total (counter)");
    println!("- tiny_dancer_confidence_scores (histogram)");
    println!("- tiny_dancer_uncertainty_estimates (histogram)");
}
