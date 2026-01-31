use serde::{Deserialize, Serialize};
use validator::Validate;

const MAX_N: usize = 100_000;
const MAX_K: usize = 1000;

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
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

fn binomial_coefficient(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }

    let k = k.min(n - k);
    let mut result = 1.0;

    for i in 0..k {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }

    result
}

fn binomial_pmf(n: usize, p: f64, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }

    let coeff = binomial_coefficient(n, k);
    let pk = p.powi(k as i32);
    let qnk = (1.0 - p).powi((n - k) as i32);

    coeff * pk * qnk
}

fn factorial(n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    (1..=n).map(|x| x as f64).product()
}

fn poisson_pmf(lambda: f64, k: usize) -> f64 {
    let lambda_k = lambda.powi(k as i32);
    let e_neg_lambda = (-lambda).exp();
    (lambda_k * e_neg_lambda) / factorial(k)
}

pub fn simulate(params: &Params) -> Vec<Result> {
    (0..params.max_k)
        .map(|k| {
            let binomial_prob = binomial_pmf(params.n, params.p, k);
            let poisson_prob = poisson_pmf(params.lambda, k);
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
    let total_variation = results.iter().map(|r| r.abs_error).sum::<f64>() * 0.5;
    let max_error = results
        .iter()
        .map(|r| r.abs_error)
        .fold(0.0, f64::max);

    let mse = results
        .iter()
        .map(|r| r.abs_error * r.abs_error)
        .sum::<f64>()
        / results.len() as f64;

    let kl_divergence = results
        .iter()
        .filter(|r| r.binomial_prob > 0.0 && r.poisson_prob > 0.0)
        .map(|r| r.binomial_prob * (r.binomial_prob / r.poisson_prob).ln())
        .sum::<f64>();

    (total_variation, max_error, mse, kl_divergence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binomial_coefficient() {
        assert!((binomial_coefficient(5, 2) - 10.0).abs() < 0.0001);
    }

    #[test]
    fn test_binomial_pmf_sum_to_one() {
        let n = 10;
        let p = 0.3;
        let sum: f64 = (0..=n).map(|k| binomial_pmf(n, p, k)).sum();
        assert!((sum - 1.0).abs() < 0.0001);
    }

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