use std::net::SocketAddr;
use std::time::Duration;
use axum::http::{Method, StatusCode};
use axum::{BoxError, Router};
use axum::error_handling::HandleErrorLayer;
use axum::routing::{get, post};
use tower::buffer::BufferLayer;
use tower::load_shed::LoadShedLayer;
use tower::ServiceBuilder;
use tower::timeout::TimeoutLayer;
use tower_http::cors::{Any, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

mod market_maker;
mod binomial_poisson;
mod brownian_motion;
mod handlers;
mod charts;
mod black_scholes;
mod utils;
mod option_strategies;

pub struct SecurityLimits;

impl SecurityLimits {
    /// Maximum request body size (5 MB)
    pub const MAX_BODY_SIZE: usize = 5 * 1024 * 1024;

    /// Request timeout (30 seconds)
    pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

    /// Rate limit: requests per minute per IP
    pub const RATE_LIMIT_PER_MINUTE: u64 = 60;

    /// Maximum concurrent requests
    pub const MAX_CONCURRENT_REQUESTS: usize = 100;
}

async fn handle_error(error: BoxError) -> (StatusCode, String) {
    if error.is::<tower::timeout::error::Elapsed>() {
        (StatusCode::REQUEST_TIMEOUT, "Request took too long".to_string())
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Unhandled internal error: {}", error))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "quant_sim=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_origin(Any)
        .allow_headers(Any);


    let app = Router::new()
        .route("/", get(handlers::serve_index))
        .route("/market-maker", get(handlers::serve_market_maker))
        .route("/glosten-milgrom", get(handlers::serve_glosten_milgrom))
        .route("/binomial-poisson", get(handlers::serve_binomial_poisson))
        .route("/brownian-motion", get(handlers::serve_brownian_motion))
        .route("/black-scholes", get(handlers::serve_black_scholes))

        .route("/simulate-glosten", post(handlers::simulate_glosten))
        .route("/simulate-binomial-poisson", post(handlers::simulate_binomial_poisson))
        .route("/simulate-brownian-motion", post(handlers::simulate_brownian_motion))
        .route("/simulate-black-scholes", post(handlers::simulate_black_scholes))
        .route("/option-strategies", get(handlers::serve_option_strategies))
        .route("/analyze-strategy", post(handlers::analyze_option_strategy))

        .nest_service("/static", ServeDir::new("static"))

        // Middleware layers
        .layer(
            ServiceBuilder::new()
                .layer(HandleErrorLayer::new(handle_error))
                .map_err(BoxError::from)
                .layer(LoadShedLayer::new())
                .layer(BufferLayer::new(1024))
                .layer(TimeoutLayer::new(SecurityLimits::REQUEST_TIMEOUT))
                .layer(RequestBodyLimitLayer::new(SecurityLimits::MAX_BODY_SIZE))
                .layer(TraceLayer::new_for_http())
        );

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    println!("Starting server on {}", addr);
    println!("Open http://localhost:8080 in your browser");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
