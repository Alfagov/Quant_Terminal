use rand::distr::Distribution;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use validator::Validate;
use rand_chacha::rand_core::SeedableRng;
use rand_distr::Normal;
use rayon::prelude::*;

const MAX_STEPS: usize = 1_000;
const MAX_PATHS: usize = 20_000;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MotionType {
    Standard,
    Geometric,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct Params {
    #[validate(range(min = 0.0))]
    pub s0: f64,
    pub mu: f64,
    pub sigma: f64,
    #[validate(range(min = 0.0))]
    pub t: f64,
    #[validate(range(min = 1, max = MAX_STEPS))]
    pub steps: usize,
    #[validate(range(min = 1, max = MAX_PATHS))]
    pub num_paths: usize,
    pub motion_type: MotionType,
    pub upper_barrier: Option<f64>,
    pub lower_barrier: Option<f64>,
}

impl Params {
    pub fn validate_limits(&self) -> Result<(), String> {
        if self.steps > MAX_STEPS {
            return Err(format!("Steps cannot exceed {}", MAX_STEPS));
        }
        if self.num_paths > MAX_PATHS {
            return Err(format!("Number of paths cannot exceed {}", MAX_PATHS));
        }
        if self.sigma < 0.0 {
            return Err("Sigma must be non-negative".to_string());
        }
        if let (Some(upper), Some(lower)) = (self.upper_barrier, self.lower_barrier) {
            if lower >= upper {
                return Err("Lower barrier must be less than upper barrier".to_string());
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PathResult {
    pub times: Vec<f64>,
    pub values: Vec<f64>,
    pub max_value: f64,
    pub min_value: f64,
    pub final_value: f64,
    pub hit_upper_barrier: bool,
    pub hit_lower_barrier: bool,
    pub ends_above_upper: bool,
    pub ends_below_lower: bool,
    pub first_passage_time_upper: Option<f64>,
    pub first_passage_time_lower: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct SimulationResult {
    pub paths: Vec<PathResult>,
    pub dt: f64,
    pub mean_final_value: f64,
    pub std_final_value: f64,
    pub mean_max_value: f64,
    pub mean_min_value: f64,
    pub prob_hit_upper: f64,
    pub prob_hit_lower: f64,
    pub prob_end_above_upper: f64,
    pub prob_end_below_lower: f64,
    pub mean_return: f64,
    pub volatility_realized: f64,
}

fn simulate_path(params: &Params) -> PathResult {
    let mut rng = ChaCha8Rng::from_rng(&mut rand::rng());
    let dt = params.t / params.steps as f64;
    let sqrt_dt = dt.sqrt();
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut times = Vec::with_capacity(params.steps + 1);
    let mut values = Vec::with_capacity(params.steps + 1);
    let mut current_value = params.s0;
    let mut hit_upper = false;
    let mut hit_lower = false;
    let mut fpt_upper = None;
    let mut fpt_lower = None;

    times.push(0.0);
    values.push(params.s0);

    for i in 1..=params.steps {
        let dw = normal.sample(&mut rng) * sqrt_dt;

        current_value = match params.motion_type {
            MotionType::Standard => {
                current_value + params.mu * dt + params.sigma * dw
            }
            MotionType::Geometric => {
                current_value
                    * ((params.mu - 0.5 * params.sigma * params.sigma) * dt
                    + params.sigma * dw)
                    .exp()
            }
        };

        let current_time = i as f64 * dt;

        // Check barriers
        if let Some(barrier) = params.upper_barrier {
            if current_value >= barrier && !hit_upper {
                hit_upper = true;
                fpt_upper = Some(current_time);
            }
        }

        if let Some(barrier) = params.lower_barrier {
            if current_value <= barrier && !hit_lower {
                hit_lower = true;
                fpt_lower = Some(current_time);
            }
        }

        times.push(current_time);
        values.push(current_value);
    }

    let final_val = *values.last().unwrap();
    let max_value = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_value = values.iter().cloned().fold(f64::INFINITY, f64::min);

    let ends_above = params
        .upper_barrier
        .map(|b| final_val >= b)
        .unwrap_or(false);

    let ends_below = params
        .lower_barrier
        .map(|b| final_val <= b)
        .unwrap_or(false);

    PathResult {
        times,
        values,
        max_value,
        min_value,
        final_value: final_val,
        hit_upper_barrier: hit_upper,
        hit_lower_barrier: hit_lower,
        ends_above_upper: ends_above,
        ends_below_lower: ends_below,
        first_passage_time_upper: fpt_upper,
        first_passage_time_lower: fpt_lower,
    }
}

pub fn run(params: &Params) -> SimulationResult {
    let paths: Vec<PathResult> = (0..params.num_paths)
        .into_par_iter()
        .map(|_| simulate_path(params))
        .collect();

    let n = paths.len() as f64;
    let final_values: Vec<f64> = paths.iter().map(|p| p.final_value).collect();

    let mean_final = final_values.iter().sum::<f64>() / n;
    let variance_final = final_values
        .iter()
        .map(|x| (x - mean_final).powi(2))
        .sum::<f64>()
        / n;
    let std_final = variance_final.sqrt();

    let mean_max = paths.iter().map(|p| p.max_value).sum::<f64>() / n;
    let mean_min = paths.iter().map(|p| p.min_value).sum::<f64>() / n;

    let prob_hit_upper = paths.iter().filter(|p| p.hit_upper_barrier).count() as f64 / n;
    let prob_hit_lower = paths.iter().filter(|p| p.hit_lower_barrier).count() as f64 / n;
    let prob_end_above = paths.iter().filter(|p| p.ends_above_upper).count() as f64 / n;
    let prob_end_below = paths.iter().filter(|p| p.ends_below_lower).count() as f64 / n;

    let mean_return = (mean_final - params.s0) / params.s0;
    let returns: Vec<f64> = final_values
        .iter()
        .map(|v| (v - params.s0) / params.s0)
        .collect();
    let mean_ret = returns.iter().sum::<f64>() / n;
    let var_ret = returns
        .iter()
        .map(|r| (r - mean_ret).powi(2))
        .sum::<f64>()
        / n;
    let vol_realized = var_ret.sqrt();

    SimulationResult {
        paths,
        dt: params.t / params.steps as f64,
        mean_final_value: mean_final,
        std_final_value: std_final,
        mean_max_value: mean_max,
        mean_min_value: mean_min,
        prob_hit_upper,
        prob_hit_lower,
        prob_end_above_upper: prob_end_above,
        prob_end_below_lower: prob_end_below,
        mean_return,
        volatility_realized: vol_realized,
    }
}


pub fn final_value_distribution(result: &SimulationResult, num_bins: usize) -> (Vec<f64>, Vec<usize>) {
    let final_values: Vec<f64> = result.paths.iter().map(|p| p.final_value).collect();

    let min_val = final_values
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_val = final_values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let range = max_val - min_val;
    let bin_width = if range == 0.0 { 1.0 } else { range / num_bins as f64 };

    let mut bins = vec![0; num_bins];

    for &value in &final_values {
        let bin_idx = if range == 0.0 {
            0
        } else {
            ((value - min_val) / bin_width).floor() as usize
        };
        let bin_idx = bin_idx.min(num_bins - 1);
        bins[bin_idx] += 1;
    }

    let bin_centers: Vec<f64> = (0..num_bins)
        .map(|i| min_val + (i as f64 + 0.5) * bin_width)
        .collect();

    (bin_centers, bins)
}

pub fn passage_time_distribution(
    result: &SimulationResult,
    barrier_type: BarrierType,
    num_bins: usize,
) -> (Vec<f64>, Vec<usize>) {
    let times: Vec<f64> = match barrier_type {
        BarrierType::Upper => result
            .paths
            .iter()
            .filter_map(|p| p.first_passage_time_upper)
            .collect(),
        BarrierType::Lower => result
            .paths
            .iter()
            .filter_map(|p| p.first_passage_time_lower)
            .collect(),
    };

    if times.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let min_time = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_time = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let bin_width = (max_time - min_time) / num_bins as f64;

    let mut bins = vec![0; num_bins];

    for &time in &times {
        let bin_idx = ((time - min_time) / bin_width).floor() as usize;
        let bin_idx = bin_idx.min(num_bins - 1);
        bins[bin_idx] += 1;
    }

    let bin_centers: Vec<f64> = (0..num_bins)
        .map(|i| min_time + (i as f64 + 0.5) * bin_width)
        .collect();

    (bin_centers, bins)
}

#[derive(Debug, Clone, Copy)]
pub enum BarrierType {
    Upper,
    Lower,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_basic() {
        let params = Params {
            s0: 100.0,
            mu: 0.05,
            sigma: 0.2,
            t: 1.0,
            steps: 252,
            num_paths: 10,
            motion_type: MotionType::Geometric,
            upper_barrier: None,
            lower_barrier: None,
        };

        let result = run(&params);

        assert_eq!(result.paths.len(), 10);
        assert!(result.mean_final_value > 0.0);
    }

    #[test]
    fn test_barrier_detection() {
        let params = Params {
            s0: 100.0,
            mu: 0.1,
            sigma: 0.3,
            t: 1.0,
            steps: 100,
            num_paths: 100,
            motion_type: MotionType::Geometric,
            upper_barrier: Some(150.0),
            lower_barrier: Some(50.0),
        };

        let result = run(&params);

        // At least some paths should hit barriers with high volatility
        assert!(result.prob_hit_upper > 0.0 || result.prob_hit_lower > 0.0);
    }
}