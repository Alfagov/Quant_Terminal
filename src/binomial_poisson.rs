use serde::{Deserialize, Serialize};
use validator::Validate;
use rayon::prelude::*;

const MAX_N: usize = 100_000;
const MAX_K: usize = 1000;

struct LogFactorialTable {
    table: Vec<f64>,
}

impl LogFactorialTable {
    fn new(max_k: usize) -> Self {
        let mut table = Vec::with_capacity(max_k + 1);
        table.push(0.0); // ln(0!) = 0
        for i in 1..=max_k {
            table.push(table[i - 1] + (i as f64).ln());
        }
        Self { table }
    }

    #[inline]
    fn ln_factorial(&self, n: usize) -> f64 {
        self.table[n]
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Validate)]
pub struct Params {
    #[validate(range(min = 1, max = 100000))]
    pub n: usize,
    #[validate(range(min = 0.0, max = 1.0))]
    pub p: f64,
    #[validate(range(min = 0.0))]
    pub lambda: f64,
    #[validate(range(min = 1, max = 1000))]
    pub max_k: usize,
}

impl Params {
    pub fn validate_limits(&self) -> std::result::Result<(), String> {
        if self.n > MAX_N {
            return Err(format!("n cannot exceed {}", MAX_N));
        }
        if self.max_k > MAX_K {
            return Err(format!("max_k cannot exceed {}", MAX_K));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Result {
    pub k: usize,
    pub binomial_prob: f64,
    pub poisson_prob: f64,
    pub abs_error: f64,
}

fn ln_binomial_coefficient(n: usize, k: usize, lft: &LogFactorialTable) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    lft.ln_factorial(n) - lft.ln_factorial(k) - lft.ln_factorial(n - k)
}

fn binomial_pmf(n: usize, p: f64, k: usize, lft: &LogFactorialTable) -> f64 {
    if k > n {
        return 0.0;
    }
    if p <= 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    if p >= 1.0 {
        return if k == n { 1.0 } else { 0.0 };
    }

    let log_pmf = ln_binomial_coefficient(n, k, lft)
        + k as f64 * p.ln()
        + (n - k) as f64 * (1.0 - p).ln();
    log_pmf.exp()
}

fn factorial(n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    (1..=n).map(|x| x as f64).product()
}

fn poisson_pmf(lambda: f64, k: usize, lft: &LogFactorialTable) -> f64 {
    if lambda <= 0.0 {
        return if k == 0 {1.0} else {0.0};
    }
    let log_pmf = k as f64 * lambda.ln() - lambda - lft.ln_factorial(k);
    log_pmf.exp()
}

pub fn simulate(params: &Params) -> Vec<Result> {
    let table_size = params.n.max(params.max_k);
    let lft = LogFactorialTable::new(table_size);

    (0..params.max_k)
        .into_par_iter()
        .map(|k| {
            let binomial_prob = binomial_pmf(params.n, params.p, k, &lft);
            let poisson_prob = poisson_pmf(params.lambda, k, &lft);
            Result {
                k,
                binomial_prob,
                poisson_prob,
                abs_error: (binomial_prob - poisson_prob).abs(),
            }
        })
        .collect()
}

pub fn calculate_metrics(results: &[Result]) -> (f64, f64, f64, f64) {
    let (sum_abs, max_err, sum_sq, kl_sum) = results
        .par_iter()
        .fold(|| (0.0f64, 0.0f64, 0.0f64, 0.0f64),
        |(sum_abs, max_err, sum_sq, kl), r| {
            let kl_term = if r.binomial_prob > 0.0 && r.poisson_prob > 0.0 {
                r.binomial_prob * (r.binomial_prob / r.poisson_prob).ln()
            } else {
                0.0
            };

            (
                sum_abs + r.abs_error,
                f64::max(max_err, r.abs_error),
                sum_sq + r.abs_error * r.abs_error,
                kl + kl_term,
            )
        })
        .reduce(
            || (0.0, 0.0, 0.0, 0.0),
            |(a1, b1, c1, d1), (a2, b2, c2, d2)| {
                (a1 + a2, f64::max(b1, b2), c1 + c2, d1 + d2)
            }
        );

    let total_variance = sum_abs * 0.5;
    let mse = sum_sq / results.len() as f64;

    (total_variance, max_err, mse, kl_sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_for_large_n_small_p() {
        let n = 1000;
        let lambda = 5.0;
        let p = lambda / n as f64;
        let params = Params {
            n,
            p,
            lambda,
            max_k: 20,
        };
        let results = simulate(&params);
        let (tv, _, _, _) = calculate_metrics(&results);
        assert!(tv < 0.01);
    }
}