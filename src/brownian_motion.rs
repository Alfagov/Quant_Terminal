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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Validate)]
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

    let (
        sum_final,
        sum_max,
        sum_min,
        cnt_hit_upper,
        cnt_hit_lower,
        cnt_end_above,
        cnt_end_below,
    ) = paths
        .par_iter()
        .fold(
            || (0.0_f64, 0.0_f64, 0.0_f64, 0u64, 0u64, 0u64, 0u64),
            |(sf, sm, sn, hu, hl, ea, eb), p| {
                (
                    sf + p.final_value,
                    sm + p.max_value,
                    sn + p.min_value,
                    hu + p.hit_upper_barrier as u64,
                    hl + p.hit_lower_barrier as u64,
                    ea + p.ends_above_upper as u64,
                    eb + p.ends_below_lower as u64,
                )
            },
        )
        .reduce(
            || (0.0, 0.0, 0.0, 0, 0, 0, 0),
            |(a1, b1, c1, d1, e1, f1, g1), (a2, b2, c2, d2, e2, f2, g2)| {
                (a1 + a2, b1 + b2, c1 + c2, d1 + d2, e1 + e2, f1 + f2, g1 + g2)
            },
        );

    let mean_final = sum_final / n;
    let mean_max = sum_max / n;
    let mean_min = sum_min / n;
    let prob_hit_upper = cnt_hit_upper as f64 / n;
    let prob_hit_lower = cnt_hit_lower as f64 / n;
    let prob_end_above = cnt_end_above as f64 / n;
    let prob_end_below = cnt_end_below as f64 / n;

    // Second pass (parallel): variance of final values + return statistics.
    // Needs mean_final from first pass, so this is a separate reduction.
    let inv_s0 = 1.0 / params.s0;
    let (sum_var_final, sum_ret, sum_ret_sq) = paths
        .par_iter()
        .fold(
            || (0.0_f64, 0.0_f64, 0.0_f64),
            |(sv, sr, sr2), p| {
                let diff = p.final_value - mean_final;
                let ret = (p.final_value - params.s0) * inv_s0;
                (sv + diff * diff, sr + ret, sr2 + ret * ret)
            },
        )
        .reduce(
            || (0.0, 0.0, 0.0),
            |(a1, b1, c1), (a2, b2, c2)| (a1 + a2, b1 + b2, c1 + c2),
        );

    let std_final = (sum_var_final / n).sqrt();
    let mean_return = (mean_final - params.s0) * inv_s0;
    let mean_ret = sum_ret / n;
    let vol_realized = (sum_ret_sq / n - mean_ret * mean_ret).max(0.0).sqrt();

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

    let (min_val, max_val) = final_values
        .par_iter()
        .fold(
            || (f64::INFINITY, f64::NEG_INFINITY),
            |(mn, mx), &v| (f64::min(mn, v), f64::max(mx, v)),
        )
        .reduce(
            || (f64::INFINITY, f64::NEG_INFINITY),
            |(mn1, mx1), (mn2, mx2)| (f64::min(mn1, mn2), f64::max(mx1, mx2)),
        );

    let range = max_val - min_val;
    let bin_width = if range == 0.0 { 1.0 } else { range / num_bins as f64 };

    // Parallel histogram: each thread builds a local bin array, then merge
    let bins = final_values
        .par_iter()
        .fold(
            || vec![0usize; num_bins],
            |mut local_bins, &value| {
                let bin_idx = if range == 0.0 {
                    0
                } else {
                    ((value - min_val) / bin_width).floor() as usize
                };
                let bin_idx = bin_idx.min(num_bins - 1);
                local_bins[bin_idx] += 1;
                local_bins
            },
        )
        .reduce(
            || vec![0usize; num_bins],
            |mut a, b| {
                for (ai, bi) in a.iter_mut().zip(b.iter()) {
                    *ai += bi;
                }
                a
            },
        );

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
            .par_iter()
            .filter_map(|p| p.first_passage_time_upper)
            .collect(),
        BarrierType::Lower => result
            .paths
            .par_iter()
            .filter_map(|p| p.first_passage_time_lower)
            .collect(),
    };

    if times.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let (min_time, max_time) = times
        .par_iter()
        .fold(
            || (f64::INFINITY, f64::NEG_INFINITY),
            |(mn, mx), &v| (f64::min(mn, v), f64::max(mx, v)),
        )
        .reduce(
            || (f64::INFINITY, f64::NEG_INFINITY),
            |(mn1, mx1), (mn2, mx2)| (f64::min(mn1, mn2), f64::max(mx1, mx2)),
        );

    let bin_width = (max_time - min_time) / num_bins as f64;

    let bins = times
        .par_iter()
        .fold(
            || vec![0usize; num_bins],
            |mut local_bins, &time| {
                let bin_idx = ((time - min_time) / bin_width).floor() as usize;
                let bin_idx = bin_idx.min(num_bins - 1);
                local_bins[bin_idx] += 1;
                local_bins
            },
        )
        .reduce(
            || vec![0usize; num_bins],
            |mut a, b| {
                for (ai, bi) in a.iter_mut().zip(b.iter()) {
                    *ai += bi;
                }
                a
            },
        );

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