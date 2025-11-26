//! Admin server example for Tiny Dancer
//!
//! This example demonstrates how to run the admin API server for monitoring,
//! health checks, and administration of the Tiny Dancer routing system.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example admin-server --features admin-api
//! ```
//!
//! ## Endpoints
//!
//! ### Health Checks
//! - `GET /health` - Basic liveness probe
//! - `GET /health/ready` - Readiness check (K8s compatible)
//!
//! ### Metrics
//! - `GET /metrics` - Prometheus format metrics
//!
//! ### Admin
//! - `POST /admin/reload` - Hot reload model
//! - `GET /admin/config` - Get current configuration
//! - `PUT /admin/config` - Update configuration
//! - `GET /admin/circuit-breaker` - Get circuit breaker status
//! - `POST /admin/circuit-breaker/reset` - Reset circuit breaker
//!
//! ### Info
//! - `GET /info` - System information
//!
//! ## Testing Endpoints
//!
//! ```bash
//! # Health check
//! curl http://localhost:8080/health
//!
//! # Readiness check
//! curl http://localhost:8080/health/ready
//!
//! # Metrics (Prometheus format)
//! curl http://localhost:8080/metrics
//!
//! # System info
//! curl http://localhost:8080/info
//!
//! # Reload model (requires auth if token is set)
//! curl -X POST http://localhost:8080/admin/reload \
//!   -H "Authorization: Bearer your-token-here"
//!
//! # Get configuration
//! curl http://localhost:8080/admin/config \
//!   -H "Authorization: Bearer your-token-here"
//!
//! # Circuit breaker status
//! curl http://localhost:8080/admin/circuit-breaker \
//!   -H "Authorization: Bearer your-token-here"
//! ```

use ruvector_tiny_dancer_core::api::{AdminServer, AdminServerConfig};
use ruvector_tiny_dancer_core::router::Router;
use ruvector_tiny_dancer_core::types::RouterConfig;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ruvector_tiny_dancer_core=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting Tiny Dancer Admin Server Example");

    // Create router with default configuration
    let router_config = RouterConfig {
        model_path: "./models/fastgrnn.safetensors".to_string(),
        confidence_threshold: 0.85,
        max_uncertainty: 0.15,
        enable_circuit_breaker: true,
        circuit_breaker_threshold: 5,
        enable_quantization: true,
        database_path: None,
    };

    tracing::info!("Creating router with config: {:?}", router_config);
    let router = Router::new(router_config)?;
    let router = Arc::new(router);

    // Configure admin server
    let admin_config = AdminServerConfig {
        bind_address: "127.0.0.1".to_string(),
        port: 8080,
        // Uncomment to enable authentication:
        // auth_token: Some("your-secret-token-here".to_string()),
        auth_token: None,
        enable_cors: true,
    };

    tracing::info!(
        "Starting admin server on {}:{}",
        admin_config.bind_address,
        admin_config.port
    );
    tracing::info!(
        "Authentication: {}",
        if admin_config.auth_token.is_some() {
            "enabled"
        } else {
            "disabled"
        }
    );

    // Create and start admin server
    let server = AdminServer::new(router, admin_config);

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║         Tiny Dancer Admin Server Running                      ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║ Health Check:  http://localhost:8080/health                   ║");
    println!("║ Readiness:     http://localhost:8080/health/ready             ║");
    println!("║ Metrics:       http://localhost:8080/metrics                  ║");
    println!("║ System Info:   http://localhost:8080/info                     ║");
    println!("║                                                                ║");
    println!("║ Admin API:     http://localhost:8080/admin/*                  ║");
    println!("║   - POST /admin/reload                                         ║");
    println!("║   - GET  /admin/config                                         ║");
    println!("║   - PUT  /admin/config                                         ║");
    println!("║   - GET  /admin/circuit-breaker                                ║");
    println!("║   - POST /admin/circuit-breaker/reset                          ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Start server (blocking)
    server.serve().await?;

    Ok(())
}
