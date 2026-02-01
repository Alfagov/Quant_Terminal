use askama::Template;
use axum::Form;
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::distribution::Normal;
use tokio::task::JoinError;
use validator::Validate;
use crate::charts::{create_dataset, create_dataset_with_dash, render_chart, Dataset};
use crate::{binomial_poisson, black_scholes, brownian_motion, market_maker, option_strategies};
use crate::brownian_motion::SimulationResult;

pub enum AppError {
    ValidationError(String),
    TemplateError(String),
    InternalError(String),
}

impl From<JoinError> for AppError {
    fn from(err: JoinError) -> Self {
        // We convert the JoinError to a string to store it in our InternalError
        AppError::InternalError(err.to_string())
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AppError::ValidationError(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::TemplateError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            AppError::InternalError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        (status, message).into_response()
    }
}

impl From<askama::Error> for AppError {
    fn from(err: askama::Error) -> Self {
        AppError::TemplateError(err.to_string())
    }
}

// Template structs with Askama
#[derive(Template)]
#[template(path = "chart.html")]
pub struct ChartTemplate {
    pub chart_id: String,
    pub title: String,
    pub labels: String,
    pub datasets: Vec<Dataset>,
    pub x_axis_label: String,
    pub y_axis_label: String,
    pub legend_display: bool,
}

#[derive(Template)]
#[template(path = "glosten_stats.html")]
pub struct GlostenStatsTemplate {
    pub true_value: String,
    pub true_state: String,
    pub final_spread: String,
    pub spread_pct: String,
    pub spread_class: String,
    pub final_belief: String,
    pub initial_belief: String,
    pub informed_pnl: String,
    pub noise_pnl: String,
    pub informed_count: usize,
    pub noise_count: usize,
    pub convergence_text: String,
    pub convergence_class: String,
}

#[derive(Template)]
#[template(path = "binomial_poisson_stats.html")]
pub struct BinomialPoissonStatsTemplate {
    pub n: usize,
    pub p: String,
    pub lambda: String,
    pub np_value: String,
    pub total_variation: String,
    pub max_error: String,
    pub mse: String,
    pub kl_divergence: String,
    pub convergence_quality: String,
    pub quality_class: String,
}

#[derive(Template)]
#[template(path = "brownian_stats.html")]
pub struct BrownianStatsTemplate {
    pub s0: String,
    pub mu: String,
    pub sigma: String,
    pub t: String,
    pub motion_type: String,
    pub mean_final: String,
    pub std_final: String,
    pub ci_lower: String,
    pub ci_upper: String,
    pub mean_max: String,
    pub mean_min: String,
    pub mean_return: String,
    pub vol_realized: String,
    pub prob_upper: String,
    pub prob_lower: String,
    pub prob_ends_upper: String,
    pub prob_ends_lower: String,
    pub has_upper_barrier: bool,
    pub has_lower_barrier: bool,
    pub upper_barrier_value: String,
    pub lower_barrier_value: String,
}

#[derive(Template)]
#[template(path = "black_scholes_stats.html")]
pub struct BlackScholesStatsTemplate {
    pub price: String,
    pub option_type: String,
    pub intrinsic_value: String,
    pub time_value: String,
    pub moneyness: String,
    pub moneyness_ratio: String,
    pub delta: String,
    pub gamma: String,
    pub theta: String,
    pub vega: String,
    pub rho: String,
    pub parity_status: String,
    pub parity_class: String,
    pub parity_residual: String,
    pub d1: String,
    pub d2: String,
    pub nd1: String,
    pub nd2: String,
}

#[derive(Template)]
#[template(path = "simulation_results.html")]
pub struct SimulationResultsTemplate {
    pub stats_html: String,
    pub charts: Vec<String>,
    pub greeks_charts: Vec<String>,
}

pub struct LegDisplay {
    pub index: usize,
    pub option_type: String,
    pub strike: String,
    pub quantity: String,
    pub price: String,
    pub delta: String,
    pub gamma: String,
}

#[derive(Template)]
#[template(path = "option_strategies_stats.html")]
pub struct OptionStrategiesStatsTemplate {
    pub net_premium: String,
    pub net_premium_class: String,
    pub premium_label: String,
    pub max_profit: String,
    pub max_loss: String,
    pub breakevens: String,
    pub breakeven_count: usize,
    pub net_delta: String,
    pub net_gamma: String,
    pub net_theta: String,
    pub net_vega: String,
    pub net_rho: String,
    pub legs: Vec<LegDisplay>,
}

// Static file handlers
pub async fn serve_index() -> Result<Html<String>, AppError> {
    serve_static_file("index.html").await
}

pub async fn serve_market_maker() -> Result<Html<String>, AppError> {
    serve_static_file("market-maker.html").await
}

pub async fn serve_glosten_milgrom() -> Result<Html<String>, AppError> {
    serve_static_file("glosten-milgrom.html").await
}

pub async fn serve_binomial_poisson() -> Result<Html<String>, AppError> {
    serve_static_file("binomial-poisson.html").await
}

pub async fn serve_brownian_motion() -> Result<Html<String>, AppError> {
    serve_static_file("brownian-motion.html").await
}

pub async fn serve_black_scholes() -> Result<Html<String>, AppError> {
    serve_static_file("black-scholes.html").await
}

pub async fn serve_option_strategies() -> Result<Html<String>, AppError> {
    serve_static_file("option-strategies.html").await
}

async fn serve_static_file(filename: &str) -> Result<Html<String>, AppError> {
    let path = format!("static/{}", filename);
    tokio::fs::read_to_string(&path)
        .await
        .map(Html)
        .map_err(|e| AppError::InternalError(format!("Failed to read file: {}", e)))
}

#[derive(Debug, Deserialize)]
pub struct BlackScholesForm {
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    option_type: String,
    num_points: usize,
}

pub async fn simulate_black_scholes(
    Form(form): Form<BlackScholesForm>,
) -> Result<Html<String>, AppError> {
    let option_type = if form.option_type == "put" {
        black_scholes::OptionType::Put
    } else {
        black_scholes::OptionType::Call
    };

    let norm = Normal::new(0.0, 1.0).unwrap();


    let params = black_scholes::Params {
        s: form.s,
        k: form.k,
        t: form.t,
        r: form.r,
        sigma: form.sigma,
        option_type,
    };

    params
        .validate()
        .map_err(|e| AppError::ValidationError(e.to_string()))?;
    params
        .validate_limits()
        .map_err(AppError::ValidationError)?;

    let num_points = form.num_points;

    let (bs_price, gs, surfaces) = tokio::task::spawn_blocking(move || {
        let pr = black_scholes::price(&params, &norm);
        let gs = black_scholes::greeks(&params, &norm);
        let surfaces = black_scholes::generate_surfaces(&params, num_points);
        (pr, gs, surfaces)
    })
        .await?;

    // Compute display values
    let intrinsic = match option_type {
        black_scholes::OptionType::Call => (form.s - form.k).max(0.0),
        black_scholes::OptionType::Put => (form.k - form.s).max(0.0),
    };
    let tv = (bs_price - intrinsic).max(0.0);

    let moneyness_ratio = form.s / form.k;
    let moneyness = if (moneyness_ratio - 1.0).abs() < 0.02 {
        "ATM"
    } else if (option_type == black_scholes::OptionType::Call && form.s > form.k)
        || (option_type == black_scholes::OptionType::Put && form.s < form.k)
    {
        "ITM"
    } else {
        "OTM"
    };

    let option_type_str = match option_type {
        black_scholes::OptionType::Call => "European Call",
        black_scholes::OptionType::Put => "European Put",
    };

    let parity_residual =
        black_scholes::put_call_parity_residual(form.s, form.k, form.t, form.r, form.sigma);
    let (parity_status, parity_class) = if parity_residual.abs() < 1e-8 {
        ("Satisfied", "text-emerald-400")
    } else {
        ("Violated", "text-red-400")
    };

    // d1, d2 for display
    let sigma_sqrt_t = form.sigma * form.t.sqrt();
    let d1_val = ((form.s / form.k).ln() + (form.r + 0.5 * form.sigma * form.sigma) * form.t)
        / sigma_sqrt_t;
    let d2_val = d1_val - sigma_sqrt_t;

    // Φ(d) approximation for display
    fn norm_cdf_display(x: f64) -> f64 {
        use std::f64::consts::SQRT_2;
        fn erf_inner(x: f64) -> f64 {
            let sign = if x >= 0.0 { 1.0 } else { -1.0 };
            let x = x.abs();
            let p = 0.3275911;
            let a1 = 0.254829592;
            let a2 = -0.284496736;
            let a3 = 1.421413741;
            let a4 = -1.453152027;
            let a5 = 1.061405429;
            let t = 1.0 / (1.0 + p * x);
            let t2 = t * t;
            let t3 = t2 * t;
            let t4 = t3 * t;
            let t5 = t4 * t;
            let y = 1.0
                - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * (-x * x).exp();
            sign * y
        }
        0.5 * (1.0 + erf_inner(x / SQRT_2))
    }

    let nd1 = norm_cdf_display(d1_val);
    let nd2 = norm_cdf_display(d2_val);

    let stats_template = BlackScholesStatsTemplate {
        price: format!("{:.4}", bs_price),
        option_type: option_type_str.to_string(),
        intrinsic_value: format!("{:.4}", intrinsic),
        time_value: format!("{:.4}", tv),
        moneyness: moneyness.to_string(),
        moneyness_ratio: format!("{:.4}", moneyness_ratio),
        delta: format!("{:.4}", gs.delta),
        gamma: format!("{:.6}", gs.gamma),
        theta: format!("{:.4}", gs.theta / 365.0),
        vega: format!("{:.4}", gs.vega / 100.0),
        rho: format!("{:.4}", gs.rho / 100.0),
        parity_status: parity_status.to_string(),
        parity_class: parity_class.to_string(),
        parity_residual: format!("{:.2e}", parity_residual),
        d1: format!("{:.4}", d1_val),
        d2: format!("{:.4}", d2_val),
        nd1: format!("{:.4}", nd1),
        nd2: format!("{:.4}", nd2),
    };

    let stats_html = stats_template.render()?;

    // ── Build charts ──────────────────────────────────────────────────────
    let spot_labels: Vec<i32> = surfaces.spot_range.iter().map(|s| *s as i32).collect();
    let vol_labels: Vec<String> = surfaces
        .vol_range
        .iter()
        .map(|v| format!("{:.0}%", v * 100.0))
        .collect();
    let time_labels: Vec<String> = surfaces
        .time_range
        .iter()
        .map(|t| format!("{:.2}", t))
        .collect();

    let charts_vec = vec![
        // 1. Price + Intrinsic + Time Value vs Spot
        render_chart(
            "priceChart",
            "Option Price vs Spot Price",
            &spot_labels,
            vec![
                create_dataset(
                    "Option Price",
                    &surfaces.prices,
                    "#10b981",
                    "rgba(16, 185, 129, 0.1)",
                    3,
                    true,
                ),
                create_dataset_with_dash(
                    "Intrinsic Value",
                    &surfaces.intrinsic,
                    "#8b5cf6",
                    "rgba(139, 92, 246, 0.0)",
                    2,
                    false,
                    vec![5, 5],
                ),
                create_dataset(
                    "Time Value",
                    &surfaces.time_value,
                    "#f59e0b",
                    "rgba(245, 158, 11, 0.1)",
                    2,
                    true,
                ),
            ],
            "Spot Price ($)",
            "Value ($)",
            true,
        )?,
    ];

    let greeks_charts_vec = vec![
        // 2. Delta vs Spot
        render_chart(
            "deltaChart",
            "Delta (\u{0394}) vs Spot Price",
            &spot_labels,
            vec![create_dataset(
                "Delta",
                &surfaces.deltas,
                "#10b981",
                "rgba(16, 185, 129, 0.1)",
                2,
                true,
            )],
            "Spot Price ($)",
            "Delta",
            false,
        )?,
        // 3. Gamma vs Spot
        render_chart(
            "gammaChart",
            "Gamma (\u{0393}) vs Spot Price",
            &spot_labels,
            vec![create_dataset(
                "Gamma",
                &surfaces.gammas,
                "#06b6d4",
                "rgba(6, 182, 212, 0.1)",
                2,
                true,
            )],
            "Spot Price ($)",
            "Gamma",
            false,
        )?,
        // 4. Theta vs Spot
        render_chart(
            "thetaChart",
            "Theta (\u{0398}) vs Spot Price \u{2014} Per Day",
            &spot_labels,
            vec![create_dataset(
                "Theta (per day)",
                &surfaces.thetas,
                "#f59e0b",
                "rgba(245, 158, 11, 0.1)",
                2,
                true,
            )],
            "Spot Price ($)",
            "Theta ($/day)",
            false,
        )?,
        // 5. Vega vs Spot
        render_chart(
            "vegaChart",
            "Vega (\u{03BD}) vs Spot Price \u{2014} Per 1% Vol",
            &spot_labels,
            vec![create_dataset(
                "Vega (per 1% vol)",
                &surfaces.vegas,
                "#8b5cf6",
                "rgba(139, 92, 246, 0.1)",
                2,
                true,
            )],
            "Spot Price ($)",
            "Vega",
            false,
        )?,
        // 6. Rho vs Spot
        render_chart(
            "rhoChart",
            "Rho (\u{03C1}) vs Spot Price \u{2014} Per 1% Rate",
            &spot_labels,
            vec![create_dataset(
                "Rho (per 1% rate)",
                &surfaces.rhos,
                "#3b82f6",
                "rgba(59, 130, 246, 0.1)",
                2,
                true,
            )],
            "Spot Price ($)",
            "Rho",
            false,
        )?,
        // 7. Price vs Volatility
        render_chart(
            "priceVolChart",
            "Option Price vs Implied Volatility",
            &vol_labels,
            vec![create_dataset(
                "Price",
                &surfaces.price_vs_vol,
                "#10b981",
                "rgba(16, 185, 129, 0.1)",
                2,
                true,
            )],
            "Volatility",
            "Price ($)",
            false,
        )?,
        // 8. Vega vs Volatility
        render_chart(
            "vegaVolChart",
            "Vega (\u{03BD}) vs Implied Volatility",
            &vol_labels,
            vec![create_dataset(
                "Vega",
                &surfaces.vega_vs_vol,
                "#8b5cf6",
                "rgba(139, 92, 246, 0.1)",
                2,
                true,
            )],
            "Volatility",
            "Vega",
            false,
        )?,
        // 9. Price vs Time to Expiry (time decay)
        render_chart(
            "priceTimeChart",
            "Option Price vs Time to Expiry (Time Decay)",
            &time_labels,
            vec![create_dataset(
                "Price",
                &surfaces.price_vs_time,
                "#10b981",
                "rgba(16, 185, 129, 0.1)",
                2,
                true,
            )],
            "Time to Expiry (years)",
            "Price ($)",
            false,
        )?,
        // 10. Theta vs Time
        render_chart(
            "thetaTimeChart",
            "Theta (\u{0398}) vs Time to Expiry \u{2014} Per Day",
            &time_labels,
            vec![create_dataset(
                "Theta (per day)",
                &surfaces.theta_vs_time,
                "#f59e0b",
                "rgba(245, 158, 11, 0.1)",
                2,
                true,
            )],
            "Time to Expiry (years)",
            "Theta ($/day)",
            false,
        )?,
    ];

    let bs_result_template = SimulationResultsTemplate {
        stats_html,
        charts: charts_vec,
        greeks_charts: greeks_charts_vec,
    };

    let html = bs_result_template.render()?;
    Ok(Html(html))
}

// Glosten-Milgrom simulation handler
#[derive(Debug, Deserialize)]
pub struct GlostenForm {
    #[serde(rename = "vH")]
    v_h: f64,
    #[serde(rename = "vL")]
    v_l: f64,
    #[serde(rename = "prior_H")]
    prior_h: f64,
    alpha: f64,
    accuracy_informed: f64,
    steps: usize,
    seed: u64,
    #[serde(rename = "true_state_h")]
    true_state_h: Option<String>,
}

pub async fn simulate_glosten(
    Form(form): Form<GlostenForm>,
) -> Result<Html<String>, AppError> {
    // Parse checkbox
    let true_state_is_high = form.true_state_h.as_deref() == Some("on");

    // Create and validate params
    let params = market_maker::Params {
        v_h: form.v_h,
        v_l: form.v_l,
        prior_h: form.prior_h,
        alpha: form.alpha,
        accuracy_informed: form.accuracy_informed,
        steps: form.steps,
        seed: form.seed,
        true_state_is_high,
    };

    params
        .validate()
        .map_err(|e| AppError::ValidationError(e.to_string()))?;
    params
        .validate_limits()
        .map_err(AppError::ValidationError)?;

    let rounds = tokio::task::spawn_blocking(move || {
        let mut sim = market_maker::Simulation::new(params.seed);
        return sim.run(&params);
    }).await?;

    // Calculate metrics
    let (informed_pnl, noise_pnl) = market_maker::cumulative_pnl(&rounds);
    let price_impacts = market_maker::get_price_impacts(&rounds);
    let returns = market_maker::calculate_returns(&rounds);
    let (informed_trades, uninformed_trades) = market_maker::separate_trades(&rounds);

    // Extract data for charts
    let len = rounds.len();
    let mut time_steps = Vec::with_capacity(len);
    let mut bids = Vec::with_capacity(len);
    let mut asks = Vec::with_capacity(len);
    let mut mids = Vec::with_capacity(len);
    let mut spreads = Vec::with_capacity(len);
    let mut beliefs = Vec::with_capacity(len);
    for r in &rounds {
        time_steps.push(r.t);
        bids.push(r.bid);
        asks.push(r.ask);
        mids.push(r.mid);
        spreads.push(r.spread);
        beliefs.push(r.belief_p_h);
    }

    let true_value = rounds.first().map(|r| r.true_value).unwrap_or(0.0);
    let final_spread = spreads.last().copied().unwrap_or(0.0);
    let final_belief = beliefs.last().copied().unwrap_or(0.0);
    let spread_pct = (final_spread / true_value) * 100.0;

    let true_state = if rounds.first().map(|r| r.true_state).unwrap_or(false) {
        "High (H)"
    } else {
        "Low (L)"
    };

    let informed_pnl_final = informed_pnl.last().copied().unwrap_or(0.0);
    let noise_pnl_final = noise_pnl.last().copied().unwrap_or(0.0).abs();

    // Create stats template
    let stats_template = GlostenStatsTemplate {
        true_value: format!("{:.2}", true_value),
        true_state: true_state.to_string(),
        final_spread: format!("{:.2}", final_spread),
        spread_pct: format!("{:.1}", spread_pct),
        spread_class: if final_spread < 5.0 {
            "text-green-600"
        } else {
            "text-orange-600"
        }
            .to_string(),
        final_belief: format!("{:.3}", final_belief),
        initial_belief: format!("{:.2}", params.prior_h),
        informed_pnl: format!("{:.2}", informed_pnl_final),
        noise_pnl: format!("{:.2}", noise_pnl_final),
        informed_count: informed_trades.len(),
        noise_count: uninformed_trades.len(),
        convergence_text: if final_spread < 1.0 { "Yes" } else { "No" }.to_string(),
        convergence_class: if final_spread < 1.0 {
            "text-green-600"
        } else {
            "text-orange-600"
        }
            .to_string(),
    };

    let stats_html = stats_template.render()?;

    // Create charts
    let charts = vec![
        render_chart(
            "priceChart",
            "Bid, Ask, and Mid Price Evolution",
            &time_steps,
            vec![
                create_dataset("Bid", &bids, "#ef4444", "rgba(239, 68, 68, 0.0)", 2, false),
                create_dataset("Ask", &asks, "#10b981", "rgba(16, 185, 129, 0.0)", 2, false),
                create_dataset("Mid Price", &mids, "#3b82f6", "rgba(59, 130, 246, 0.1)", 3, true),
                create_dataset_with_dash(
                    "True Value",
                    &vec![true_value; time_steps.len()],
                    "#8b5cf6",
                    "rgba(139, 92, 246, 0.0)",
                    2,
                    false,
                    vec![5, 5],
                ),
            ],
            "Time Step",
            "Price ($)",
            true,
        )?,
        render_chart(
            "spreadChart",
            "Bid-Ask Spread Over Time",
            &time_steps,
            vec![create_dataset(
                "Spread",
                &spreads,
                "#f59e0b",
                "rgba(245, 158, 11, 0.1)",
                2,
                true,
            )],
            "Time Step",
            "Spread ($)",
            false,
        )?,
        render_chart(
            "beliefChart",
            "Market Maker Belief P(H) Evolution",
            &time_steps,
            vec![create_dataset(
                "Belief P(H)",
                &beliefs,
                "#06b6d4",
                "rgba(6, 182, 212, 0.1)",
                2,
                true,
            )],
            "Time Step",
            "Probability",
            false,
        )?,
        render_chart(
            "pnlChart",
            "Cumulative Trader PnL",
            &time_steps,
            vec![
                create_dataset(
                    "Informed Traders PnL",
                    &informed_pnl,
                    "#10b981",
                    "rgba(16, 185, 129, 0.1)",
                    3,
                    true,
                ),
                create_dataset(
                    "Noise Traders PnL",
                    &noise_pnl,
                    "#ef4444",
                    "rgba(239, 68, 68, 0.1)",
                    3,
                    true,
                ),
            ],
            "Time Step",
            "Cumulative PnL ($)",
            true,
        )?,
        render_chart(
            "impactChart",
            "Price Impact per Trade",
            &time_steps,
            vec![create_dataset(
                "Price Impact",
                &price_impacts,
                "#8b5cf6",
                "rgba(139, 92, 246, 0.1)",
                2,
                false,
            )],
            "Time Step",
            "Price Impact ($)",
            false,
        )?,
        render_chart(
            "returnsChart",
            "Absolute Returns Distribution",
            &(0..returns.len()).collect::<Vec<_>>(),
            vec![create_dataset(
                "Absolute Returns",
                &returns.iter().map(|r| r.abs()).collect::<Vec<_>>(),
                "#ec4899",
                "rgba(236, 72, 153, 0.5)",
                1,
                true,
            )],
            "Trade Number",
            "Absolute Return",
            false,
        )?,
    ];

    let result_template = SimulationResultsTemplate {
        stats_html,
        charts,
        greeks_charts: vec![],
    };

    let html = result_template.render()?;
    Ok(Html(html))
}

// Binomial-Poisson handler
#[derive(Debug, Deserialize)]
pub struct BinomialPoissonForm {
    n: usize,
    p: f64,
    max_k: usize,
}

pub async fn simulate_binomial_poisson(
    Form(form): Form<BinomialPoissonForm>,
) -> Result<Html<String>, AppError> {
    let params = binomial_poisson::Params {
        n: form.n,
        p: form.p,
        lambda: form.n as f64 * form.p,
        max_k: form.max_k,
    };

    params
        .validate()
        .map_err(|e| AppError::ValidationError(e.to_string()))?;
    params
        .validate_limits()
        .map_err(AppError::ValidationError)?;

    let (results, (tv, max_err, mse, kl)) = tokio::task::spawn_blocking(move || {
        let results = binomial_poisson::simulate(&params);
        let metrics = binomial_poisson::calculate_metrics(&results);

        (results, metrics)
    }).await?;

    let k_values: Vec<usize> = results.iter().map(|r| r.k).collect();
    let binomial_probs: Vec<f64> = results.iter().map(|r| r.binomial_prob).collect();
    let poisson_probs: Vec<f64> = results.iter().map(|r| r.poisson_prob).collect();
    let abs_errors: Vec<f64> = results.iter().map(|r| r.abs_error).collect();

    // Stats context
    let (quality, quality_class) = if tv < 0.01 {
        ("Excellent", "text-green-600")
    } else if tv < 0.05 {
        ("Good", "text-blue-600")
    } else if tv < 0.1 {
        ("Fair", "text-orange-600")
    } else {
        ("Poor", "text-red-600")
    };

    let stats_template = BinomialPoissonStatsTemplate {
        n: params.n,
        p: format!("{:.6}", params.p),
        lambda: format!("{:.4}", params.lambda),
        np_value: format!("{:.4}", params.n as f64 * params.p),
        total_variation: format!("{:.6}", tv),
        max_error: format!("{:.6}", max_err),
        mse: format!("{:.8}", mse),
        kl_divergence: format!("{:.6}", kl),
        convergence_quality: quality.to_string(),
        quality_class: quality_class.to_string(),
    };

    let stats_html = stats_template.render()?;

    let charts = vec![
        render_chart(
            "distributionChart",
            "Binomial vs Poisson Distribution",
            &k_values,
            vec![
                create_dataset(
                    &format!("Binomial(n={}, p={:.6})", params.n, params.p),
                    &binomial_probs,
                    "#3b82f6",
                    "rgba(59, 130, 246, 0.2)",
                    2,
                    true,
                ),
                create_dataset_with_dash(
                    &format!("Poisson(λ={:.4})", params.lambda),
                    &poisson_probs,
                    "#ef4444",
                    "rgba(239, 68, 68, 0.0)",
                    3,
                    false,
                    vec![5, 5],
                ),
            ],
            "k (Number of Successes)",
            "Probability P(X = k)",
            true,
        )?,
        render_chart(
            "errorChart",
            "Absolute Error Between Distributions",
            &k_values,
            vec![create_dataset(
                "Absolute Error",
                &abs_errors,
                "#f59e0b",
                "rgba(245, 158, 11, 0.3)",
                2,
                true,
            )],
            "k (Number of Successes)",
            "Absolute Error",
            false,
        )?];
    let result_template = SimulationResultsTemplate {
        stats_html,
        charts,
        greeks_charts: vec![],
    };

    let html = result_template.render()?;
    Ok(Html(html))
}

// Option Strategies handler
#[derive(Debug, Deserialize)]
pub struct OptionStrategyForm {
    s: f64,
    t: f64,
    r: f64,
    sigma: f64,
    num_points: usize,
    legs_json: String,
}

#[derive(Debug, Deserialize)]
struct LegInput {
    option_type: String,
    strike: f64,
    quantity: i32,
}

pub async fn analyze_option_strategy(
    Form(form): Form<OptionStrategyForm>,
) -> Result<Html<String>, AppError> {
    let leg_inputs: Vec<LegInput> = serde_json::from_str(&form.legs_json)
        .map_err(|e| AppError::ValidationError(format!("Invalid legs data: {}", e)))?;

    let legs: Vec<option_strategies::Leg> = leg_inputs
        .iter()
        .map(|li| option_strategies::Leg {
            option_type: if li.option_type == "Call" {
                black_scholes::OptionType::Call
            } else {
                black_scholes::OptionType::Put
            },
            strike: li.strike,
            quantity: li.quantity,
        })
        .collect();

    let params = option_strategies::StrategyParams {
        spot: form.s,
        t: form.t,
        r: form.r,
        sigma: form.sigma,
        legs: legs.clone(),
        num_points: form.num_points,
    };

    params.validate().map_err(AppError::ValidationError)?;

    let result = tokio::task::spawn_blocking(move || option_strategies::analyze_strategy(&params))
        .await?;

    // Build per-leg display data
    let legs_display: Vec<LegDisplay> = legs
        .iter()
        .enumerate()
        .map(|(i, leg)| {
            let type_str = match leg.option_type {
                black_scholes::OptionType::Call => "Call",
                black_scholes::OptionType::Put => "Put",
            };
            LegDisplay {
                index: i + 1,
                option_type: type_str.to_string(),
                strike: format!("{:.2}", leg.strike),
                quantity: format!("{:+}", leg.quantity),
                price: format!("{:.4}", result.leg_prices[i]),
                delta: format!("{:.4}", result.leg_greeks[i].delta),
                gamma: format!("{:.6}", result.leg_greeks[i].gamma),
            }
        })
        .collect();

    let premium_is_debit = result.net_premium > 0.0;

    let stats_template = OptionStrategiesStatsTemplate {
        net_premium: format!("${:.2}", result.net_premium.abs()),
        net_premium_class: if premium_is_debit {
            "text-red-400"
        } else {
            "text-emerald-400"
        }
            .to_string(),
        premium_label: if premium_is_debit { "DEBIT" } else { "CREDIT" }.to_string(),
        max_profit: if result.max_profit > 1e6 {
            "Unlimited".to_string()
        } else {
            format!("${:.2}", result.max_profit)
        },
        max_loss: if result.max_loss < -1e6 {
            "Unlimited".to_string()
        } else {
            format!("${:.2}", result.max_loss)
        },
        breakevens: result
            .breakeven_points
            .iter()
            .map(|b| format!("${:.2}", b))
            .collect::<Vec<_>>()
            .join(", "),
        breakeven_count: result.breakeven_points.len(),
        net_delta: format!("{:.4}", result.net_greeks.delta),
        net_gamma: format!("{:.6}", result.net_greeks.gamma),
        net_theta: format!("{:.4}", result.net_greeks.theta / 365.0),
        net_vega: format!("{:.4}", result.net_greeks.vega / 100.0),
        net_rho: format!("{:.4}", result.net_greeks.rho / 100.0),
        legs: legs_display,
    };

    let stats_html = stats_template.render()?;

    let spot_labels: Vec<i32> = result.spot_range.iter().map(|s| *s as i32).collect();

    let charts = vec![
        render_chart(
            "payoffChart",
            "Strategy Payoff & PnL at Expiry",
            &spot_labels,
            vec![
                create_dataset(
                    "Payoff at Expiry",
                    &result.payoff_at_expiry,
                    "#10b981",
                    "rgba(16, 185, 129, 0.05)",
                    2,
                    false,
                ),
                create_dataset(
                    "PnL at Expiry",
                    &result.pnl_at_expiry,
                    "#3b82f6",
                    "rgba(59, 130, 246, 0.1)",
                    3,
                    true,
                ),
                create_dataset_with_dash(
                    "Breakeven",
                    &vec![0.0; spot_labels.len()],
                    "#94a3b8",
                    "rgba(0, 0, 0, 0)",
                    1,
                    false,
                    vec![5, 5],
                ),
            ],
            "Spot Price ($)",
            "Profit / Loss ($)",
            true,
        )?,
        render_chart(
            "currentValueChart",
            "Current Strategy Value vs Spot",
            &spot_labels,
            vec![create_dataset(
                "Strategy Value",
                &result.current_value,
                "#8b5cf6",
                "rgba(139, 92, 246, 0.1)",
                2,
                true,
            )],
            "Spot Price ($)",
            "Value ($)",
            false,
        )?,
    ];

    let greeks_charts = vec![
        render_chart(
            "netDeltaChart",
            "Net Delta (\u{0394}) vs Spot Price",
            &spot_labels,
            vec![create_dataset(
                "Net Delta",
                &result.net_deltas,
                "#10b981",
                "rgba(16, 185, 129, 0.1)",
                2,
                true,
            )],
            "Spot Price ($)",
            "Delta",
            false,
        )?,
        render_chart(
            "netGammaChart",
            "Net Gamma (\u{0393}) vs Spot Price",
            &spot_labels,
            vec![create_dataset(
                "Net Gamma",
                &result.net_gammas,
                "#06b6d4",
                "rgba(6, 182, 212, 0.1)",
                2,
                true,
            )],
            "Spot Price ($)",
            "Gamma",
            false,
        )?,
        render_chart(
            "netThetaChart",
            "Net Theta (\u{0398}) vs Spot Price \u{2014} Per Day",
            &spot_labels,
            vec![create_dataset(
                "Net Theta (per day)",
                &result.net_thetas,
                "#f59e0b",
                "rgba(245, 158, 11, 0.1)",
                2,
                true,
            )],
            "Spot Price ($)",
            "Theta ($/day)",
            false,
        )?,
        render_chart(
            "netVegaChart",
            "Net Vega (\u{03BD}) vs Spot Price \u{2014} Per 1% Vol",
            &spot_labels,
            vec![create_dataset(
                "Net Vega (per 1% vol)",
                &result.net_vegas,
                "#8b5cf6",
                "rgba(139, 92, 246, 0.1)",
                2,
                true,
            )],
            "Spot Price ($)",
            "Vega",
            false,
        )?,
        render_chart(
            "netRhoChart",
            "Net Rho (\u{03C1}) vs Spot Price \u{2014} Per 1% Rate",
            &spot_labels,
            vec![create_dataset(
                "Net Rho (per 1% rate)",
                &result.net_rhos,
                "#3b82f6",
                "rgba(59, 130, 246, 0.1)",
                2,
                true,
            )],
            "Spot Price ($)",
            "Rho",
            false,
        )?,
    ];

    let result_template = SimulationResultsTemplate {
        stats_html,
        charts,
        greeks_charts,
    };

    let html = result_template.render()?;
    Ok(Html(html))
}

// Brownian Motion handler
#[derive(Debug, Deserialize)]
pub struct BrownianMotionForm {
    s0: f64,
    mu: f64,
    sigma: f64,
    t: f64,
    steps: usize,
    num_paths: usize,
    motion_type: String,
    enable_upper_barrier: Option<String>,
    upper_barrier: Option<f64>,
    enable_lower_barrier: Option<String>,
    lower_barrier: Option<f64>,
}

pub async fn simulate_brownian_motion(
    Form(form): Form<BrownianMotionForm>,
) -> Result<Html<String>, AppError> {
    let motion_type = if form.motion_type == "standard" {
        brownian_motion::MotionType::Standard
    } else {
        brownian_motion::MotionType::Geometric
    };

    let upper_barrier = if form.enable_upper_barrier.as_deref() == Some("on") {
        form.upper_barrier
    } else {
        None
    };

    let lower_barrier = if form.enable_lower_barrier.as_deref() == Some("on") {
        form.lower_barrier
    } else {
        None
    };

    let params = brownian_motion::Params {
        s0: form.s0,
        mu: form.mu,
        sigma: form.sigma,
        t: form.t,
        steps: form.steps,
        num_paths: form.num_paths,
        motion_type,
        upper_barrier,
        lower_barrier,
    };

    params
        .validate()
        .map_err(|e| AppError::ValidationError(e.to_string()))?;
    params
        .validate_limits()
        .map_err(AppError::ValidationError)?;

    let result = tokio::task::spawn_blocking(move || brownian_motion::run(&params))
        .await?;

    // Generate stats
    let motion_type_str = match motion_type {
        brownian_motion::MotionType::Standard => "Standard BM",
        brownian_motion::MotionType::Geometric => "Geometric BM",
    };

    let ci_lower = result.mean_final_value - 1.96 * result.std_final_value;
    let ci_upper = result.mean_final_value + 1.96 * result.std_final_value;

    let stats_template = BrownianStatsTemplate {
        s0: format!("{:.2}", params.s0),
        mu: format!("{:.4}", params.mu),
        sigma: format!("{:.4}", params.sigma),
        t: format!("{:.2}", params.t),
        motion_type: motion_type_str.to_string(),
        mean_final: format!("{:.2}", result.mean_final_value),
        std_final: format!("{:.2}", result.std_final_value),
        ci_lower: format!("{:.2}", ci_lower),
        ci_upper: format!("{:.2}", ci_upper),
        mean_max: format!("{:.2}", result.mean_max_value),
        mean_min: format!("{:.2}", result.mean_min_value),
        mean_return: format!("{:.2}", result.mean_return * 100.0),
        vol_realized: format!("{:.2}", result.volatility_realized * 100.0),
        prob_upper: format!("{:.1}", result.prob_hit_upper * 100.0),
        prob_lower: format!("{:.1}", result.prob_hit_lower * 100.0),
        prob_ends_upper: format!("{:.1}", result.prob_end_above_upper * 100.0),
        prob_ends_lower: format!("{:.1}", result.prob_end_below_lower * 100.0),
        has_upper_barrier: upper_barrier.is_some(),
        has_lower_barrier: lower_barrier.is_some(),
        upper_barrier_value: upper_barrier
            .map(|v| format!("{:.2}", v))
            .unwrap_or_else(|| "N/A".to_string()),
        lower_barrier_value: lower_barrier
            .map(|v| format!("{:.2}", v))
            .unwrap_or_else(|| "N/A".to_string()),
    };

    let stats_html = stats_template.render()?;

    // Prepare chart data - show first 10 paths
    let paths_to_show = result.paths.iter().take(10).collect::<Vec<_>>();
    let times = if let Some(path) = result.paths.first() {
        &path.times
    } else {
        &Vec::new()
    };

    let time_labels: Vec<usize> = (0..times.len()).collect();

    // Create path datasets
    let mut path_datasets: Vec<Dataset> = paths_to_show
        .iter()
        .enumerate()
        .map(|(i, path)| {
            let color_idx = i % 6;
            let colors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"];
            create_dataset(
                &format!("Path {}", i + 1),
                &path.values,
                colors[color_idx],
                "rgba(0, 0, 0, 0.0)",
                1,
                false,
            )
        })
        .collect();

    let num_time_steps = times.len();
    let num_paths = result.paths.len();
    let inv_n = 1.0 / num_paths as f64;

    // Parallel over time steps: each iteration reads column i from all paths,
    // computes mean and std in a single pass, and produces (mean, upper_ci, lower_ci).
    let (mean_path, (upper_ci, lower_ci)): (Vec<f64>, (Vec<f64>, Vec<f64>)) = (0..num_time_steps)
        .into_par_iter()
        .map(|i| {
            let (sum, sum_sq) = result
                .paths
                .iter()
                .fold((0.0_f64, 0.0_f64), |(s, sq), p| {
                    let v = p.values.get(i).copied().unwrap_or(0.0);
                    (s + v, sq + v * v)
                });
            let mean = sum * inv_n;
            // Var(X) = E[X²] - E[X]² (computational formula, single pass)
            let variance = (sum_sq * inv_n - mean * mean).max(0.0);
            let ci_offset = 1.96 * variance.sqrt();
            (mean, (mean + ci_offset, mean - ci_offset))
        })
        .unzip();

    path_datasets.push(create_dataset(
        "Mean Path",
        &mean_path,
        "#000000",
        "rgba(0, 0, 0, 0.0)",
        3,
        false,
    ));

    path_datasets.push(create_dataset_with_dash(
        "95% CI Upper",
        &upper_ci,
        "#94a3b8",
        "rgba(148, 163, 184, 0.1)",
        1,
        false,
        vec![5, 5],
    ));

    path_datasets.push(create_dataset_with_dash(
        "95% CI Lower",
        &lower_ci,
        "#94a3b8",
        "rgba(148, 163, 184, 0.1)",
        1,
        false,
        vec![5, 5],
    ));

    // Add barrier lines if enabled
    if let Some(barrier) = upper_barrier {
        path_datasets.push(create_dataset_with_dash(
            "Upper Barrier",
            &vec![barrier; time_labels.len()],
            "#dc2626",
            "rgba(220, 38, 38, 0.0)",
            2,
            false,
            vec![10, 5],
        ));
    }

    if let Some(barrier) = lower_barrier {
        path_datasets.push(create_dataset_with_dash(
            "Lower Barrier",
            &vec![barrier; time_labels.len()],
            "#dc2626",
            "rgba(220, 38, 38, 0.0)",
            2,
            false,
            vec![10, 5],
        ));
    }

    let y_label = match motion_type {
        brownian_motion::MotionType::Standard => "Value",
        brownian_motion::MotionType::Geometric => "Price ($)",
    };

    let mut charts = vec![
        render_chart(
            "pathsChart",
            &format!(
                "Brownian Motion Paths ({} of {} shown)",
                paths_to_show.len().min(10),
                params.num_paths
            ),
            &time_labels,
            path_datasets,
            "Time Step",
            y_label,
            false,
        )?,
    ];

    // Final value distribution
    let (bin_centers, counts) = brownian_motion::final_value_distribution(&result, 30);
    let bin_labels: Vec<i32> = bin_centers.iter().map(|c| *c as i32).collect();
    let count_floats: Vec<f64> = counts.iter().map(|c| *c as f64).collect();

    let x_label = match motion_type {
        brownian_motion::MotionType::Standard => "Final Value",
        brownian_motion::MotionType::Geometric => "Final Price ($)",
    };

    charts.push(render_chart(
        "distributionChart",
        "Distribution of Final Values",
        &bin_labels,
        vec![create_dataset(
            "Final Value Distribution",
            &count_floats,
            "#3b82f6",
            "rgba(59, 130, 246, 0.5)",
            1,
            true,
        )],
        x_label,
        "Frequency",
        false,
    )?);

    // First passage time charts
    if upper_barrier.is_some() {
        let (fpt_times, fpt_counts) = brownian_motion::passage_time_distribution(
            &result,
            brownian_motion::BarrierType::Upper,
            20,
        );
        if !fpt_times.is_empty() {
            let fpt_labels: Vec<i32> = fpt_times.iter().map(|t| (t * 100.0) as i32).collect();
            let fpt_floats: Vec<f64> = fpt_counts.iter().map(|c| *c as f64).collect();

            charts.push(render_chart(
                "fptUpperChart",
                "First Passage Time to Upper Barrier",
                &fpt_labels,
                vec![create_dataset(
                    "First Passage Time",
                    &fpt_floats,
                    "#10b981",
                    "rgba(16, 185, 129, 0.5)",
                    1,
                    true,
                )],
                "Time",
                "Frequency",
                false,
            )?);
        }
    }

    if lower_barrier.is_some() {
        let (fpt_times, fpt_counts) = brownian_motion::passage_time_distribution(
            &result,
            brownian_motion::BarrierType::Lower,
            20,
        );
        if !fpt_times.is_empty() {
            let fpt_labels: Vec<i32> = fpt_times.iter().map(|t| (t * 100.0) as i32).collect();
            let fpt_floats: Vec<f64> = fpt_counts.iter().map(|c| *c as f64).collect();

            charts.push(render_chart(
                "fptLowerChart",
                "First Passage Time to Lower Barrier",
                &fpt_labels,
                vec![create_dataset(
                    "First Passage Time",
                    &fpt_floats,
                    "#ef4444",
                    "rgba(239, 68, 68, 0.5)",
                    1,
                    true,
                )],
                "Time",
                "Frequency",
                false,
            )?);
        }
    }

    let result_template = SimulationResultsTemplate {
        stats_html,
        charts,
        greeks_charts: vec![],
    };

    let html = result_template.render()?;
    Ok(Html(html))
}