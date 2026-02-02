use avellaneda_stoikov_rs::model::{ExponentialIntensity, Parameters};
use avellaneda_stoikov_rs::sim::{SimConfig, SimResult, run_trajectory};
use serde::{Deserialize, Serialize};
use validator::Validate;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Validate)]
pub struct Params {
    #[validate(range(min = 0.0, max = 100.0))]
    pub gamma: f64, // Risk aversion
    #[validate(range(min = 0.0, max = 10.0))]
    pub sigma: f64, // Volatility
    #[validate(range(min = 0.0, max = 30.0))]
    pub drift: f64, // Volatility
    #[validate(range(min = 0.1, max = 10.0))]
    pub t: f64, // Time horizon (T)
    #[validate(range(min = 0.01, max = 100.0))]
    pub k: f64, // Intensity param k
    #[validate(range(min = 0.01, max = 1000.0))]
    pub a: f64, // Intensity param A
    #[validate(range(min = 100, max = 10000))]
    pub steps: usize,
    #[validate(range(min = 10.0, max = 10000.0))]
    pub s0: f64,
}

#[derive(Debug, Serialize)]
pub struct SimulationOutput {
    pub time_steps: Vec<f64>,
    pub mid_prices: Vec<f64>,
    pub inventory: Vec<i32>,
    pub bid_prices: Vec<f64>,
    pub ask_prices: Vec<f64>,
    pub wealth: Vec<f64>,
    pub final_pnl: f64,
}

pub fn simulate(params: &Params) -> SimulationOutput {
    let model_params = Parameters {
        gamma: params.gamma,
        sigma: params.sigma,
        t_horizon: params.t,
        k: params.k,
        a: params.a,
    };

    let dt = params.t / params.steps as f64;
    let sim_config = SimConfig {
        dt,
        num_steps: params.steps,
        s_0: params.s0,
        drift: params.drift,
        latency_steps: 0,
    };

    // Use exponential intensity model
    let intensity_model = ExponentialIntensity {
        k: params.k,
        a: params.a,
    };

    let result: SimResult = run_trajectory(&model_params, &sim_config, &intensity_model);

    // Extract data for charts
    let mut time_steps = Vec::with_capacity(result.trajectory.len());
    let mut mid_prices = Vec::with_capacity(result.trajectory.len());
    let mut inventory = Vec::with_capacity(result.trajectory.len());
    let mut bid_prices = Vec::with_capacity(result.trajectory.len());
    let mut ask_prices = Vec::with_capacity(result.trajectory.len());
    let mut wealth = Vec::with_capacity(result.trajectory.len());

    for step in result.trajectory {
        time_steps.push(step.time);
        mid_prices.push(step.mid_price);
        inventory.push(step.inventory);
        bid_prices.push(step.bid_price);
        ask_prices.push(step.ask_price);
        wealth.push(step.wealth);
    }

    SimulationOutput {
        time_steps,
        mid_prices,
        inventory,
        bid_prices,
        ask_prices,
        wealth,
        final_pnl: result.final_pnl,
    }
}
