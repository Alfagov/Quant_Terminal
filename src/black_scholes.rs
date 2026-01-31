use serde::{Deserialize, Serialize};
use statrs::distribution::{Continuous, ContinuousCDF, Normal};
use validator::Validate;
use rayon::prelude::*;
use crate::utils::{unzip3, unzip8};

const MAX_SURFACE_POINTS: usize = 200;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OptionType {
    Call,
    Put,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Validate)]
pub struct Params {
    #[validate(range(min = 0.01, max = 10000.0))]
    pub s: f64,

    #[validate(range(min = 0.01, max = 10000.0))]
    pub k: f64,

    #[validate(range(min = 0.001, max = 30.0))]
    pub t: f64,

    pub r: f64,

    #[validate(range(min = 0.001, max = 5.0))]
    pub sigma: f64,

    pub option_type: OptionType,
}

impl Params {
    pub fn validate_limits(&self) -> Result<(), String> {
        if self.sigma <= 0.0 {
            return Err("Volatility must be positive".to_string());
        }
        if self.t <= 0.0 {
            return Err("Time to expiration must be positive".to_string());
        }
        if self.s <= 0.0 {
            return Err("Spot price must be positive".to_string());
        }
        if self.k <= 0.0 {
            return Err("Strike price must be positive".to_string());
        }
        Ok(())
    }
}

fn d1_d2(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> (f64, f64) {
    let sigma_sqrt_t = sigma * t.sqrt();
    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / sigma_sqrt_t;
    let d2 = d1 - sigma_sqrt_t;
    (d1, d2)
}

/// Black-Scholes option price.
pub fn price(p: &Params, n: &Normal) -> f64 {
    let (d1, d2) = d1_d2(p.s, p.k, p.t, p.r, p.sigma);
    let df = (-p.r * p.t).exp(); // discount factor e^{-rT}

    match p.option_type {
        OptionType::Call => p.s * n.cdf(d1) - p.k * df * n.cdf(d2),
        OptionType::Put => p.k * df * n.cdf(-d2) - p.s * n.cdf(-d1),
    }
}

/// Δ (Delta): ∂V/∂S — sensitivity to the underlying price.
pub fn delta(p: &Params, n: &Normal) -> f64 {
    let (d1, _) = d1_d2(p.s, p.k, p.t, p.r, p.sigma);
    match p.option_type {
        OptionType::Call => n.cdf(d1),
        OptionType::Put => n.cdf(d1) - 1.0,
    }
}

/// Γ (Gamma): ∂²V/∂S² — rate of change of delta.
pub fn gamma(p: &Params, n: &Normal) -> f64 {
    let (d1, _) = d1_d2(p.s, p.k, p.t, p.r, p.sigma);
    n.pdf(d1) / (p.s * p.sigma * p.t.sqrt())
}

/// Θ (Theta): ∂V/∂t — sensitivity to passage of time (per year).
pub fn theta(p: &Params, n: &Normal) -> f64 {
    let (d1, d2) = d1_d2(p.s, p.k, p.t, p.r, p.sigma);
    let df = (-p.r * p.t).exp();
    let term1 = -(p.s * n.pdf(d1) * p.sigma) / (2.0 * p.t.sqrt());

    match p.option_type {
        OptionType::Call => term1 - p.r * p.k * df * n.cdf(d2),
        OptionType::Put => term1 + p.r * p.k * df * n.cdf(-d2),
    }
}

/// ν (Vega): ∂V/∂σ — sensitivity to volatility.
pub fn vega(p: &Params, n: &Normal) -> f64 {
    let (d1, _) = d1_d2(p.s, p.k, p.t, p.r, p.sigma);
    p.s * n.pdf(d1) * p.t.sqrt()
}

/// ρ (Rho): ∂V/∂r — sensitivity to the risk-free rate.
pub fn rho(p: &Params, n: &Normal) -> f64 {
    let (_, d2) = d1_d2(p.s, p.k, p.t, p.r, p.sigma);
    let df = (-p.r * p.t).exp();
    match p.option_type {
        OptionType::Call => p.k * p.t * df * n.cdf(d2),
        OptionType::Put => -p.k * p.t * df * n.cdf(-d2),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub rho: f64,
}

pub fn greeks(p: &Params, n: &Normal) -> Greeks {
    Greeks {
        delta: delta(p, n),
        gamma: gamma(p, n),
        theta: theta(p, n),
        vega: vega(p, n),
        rho: rho(p, n),
    }
}

#[derive(Debug, Serialize)]
pub struct SurfaceResult {
    /// Spot prices used for the sweep
    pub spot_range: Vec<f64>,
    /// Price curve across spot
    pub prices: Vec<f64>,
    /// Delta curve across spot
    pub deltas: Vec<f64>,
    /// Gamma curve across spot
    pub gammas: Vec<f64>,
    /// Theta curve across spot (per day, divided by 365)
    pub thetas: Vec<f64>,
    /// Vega curve across spot (per 1% vol point, divided by 100)
    pub vegas: Vec<f64>,
    /// Rho curve across spot (per 1% rate point, divided by 100)
    pub rhos: Vec<f64>,
    /// Intrinsic value across spot (payoff at expiration)
    pub intrinsic: Vec<f64>,
    /// Time value across spot (price − intrinsic)
    pub time_value: Vec<f64>,

    // Volatility smile: price/Greeks across vol at fixed spot
    pub vol_range: Vec<f64>,
    pub price_vs_vol: Vec<f64>,
    pub delta_vs_vol: Vec<f64>,
    pub vega_vs_vol: Vec<f64>,

    // Time decay: price/Greeks across T at fixed spot
    pub time_range: Vec<f64>,
    pub price_vs_time: Vec<f64>,
    pub delta_vs_time: Vec<f64>,
    pub theta_vs_time: Vec<f64>,
}

pub fn generate_surfaces(base: &Params, num_points: usize) -> SurfaceResult {
    let n = num_points.min(MAX_SURFACE_POINTS);
    let norm = Normal::new(0.0, 1.0).unwrap();

    let s_min = (base.k * 0.5).max(0.01);
    let s_max = base.k * 1.5;
    let ds = (s_max - s_min) / (n - 1) as f64;
    let spot_range: Vec<f64> = (0..n).map(|i| s_min + i as f64 * ds).collect();

    let spot_results: Vec<(f64, f64, f64, f64, f64, f64, f64, f64)> = spot_range
        .par_iter()
        .map(|&s| {
            let p = Params { s, ..*base };
            let pr = price(&p, &norm);
            let d = delta(&p, &norm);
            let g = gamma(&p, &norm);
            let th = theta(&p, &norm) / 365.0; // per-day theta
            let v = vega(&p, &norm) / 100.0;   // per 1% vol
            let r = rho(&p, &norm) / 100.0;    // per 1% rate
            let intrinsic = match base.option_type {
                OptionType::Call => (s - base.k).max(0.0),
                OptionType::Put => (base.k - s).max(0.0),
            };
            let tv = (pr - intrinsic).max(0.0);
            (pr, d, g, th, v, r, intrinsic, tv)
        })
        .collect();

    let (prices, deltas, gammas, thetas, vegas, rhos, intrinsic, time_value) =
        unzip8(spot_results);

    let vol_min = 0.05;
    let vol_max = 1.5;
    let dv = (vol_max - vol_min) / (n - 1) as f64;
    let vol_range: Vec<f64> = (0..n).map(|i| vol_min + i as f64 * dv).collect();

    let vol_results: Vec<(f64, f64, f64)> = vol_range
        .par_iter()
        .map(|&sigma| {
            let p = Params { sigma, ..*base };
            (price(&p, &norm), delta(&p, &norm), vega(&p, &norm) / 100.0)
        })
        .collect();

    let (price_vs_vol, delta_vs_vol, vega_vs_vol) = unzip3(vol_results);

    let t_min = 0.01; // ~4 days
    let t_max = base.t.max(0.02);
    let dt = (t_max - t_min) / (n - 1) as f64;
    let time_range: Vec<f64> = (0..n).map(|i| t_min + i as f64 * dt).collect();

    let time_results: Vec<(f64, f64, f64)> = time_range
        .par_iter()
        .map(|&t| {
            let p = Params { t, ..*base };
            (price(&p, &norm), delta(&p, &norm), theta(&p, &norm) / 365.0)
        })
        .collect();

    let (price_vs_time, delta_vs_time, theta_vs_time) = unzip3(time_results);

    SurfaceResult {
        spot_range,
        prices,
        deltas,
        gammas,
        thetas,
        vegas,
        rhos,
        intrinsic,
        time_value,
        vol_range,
        price_vs_vol,
        delta_vs_vol,
        vega_vs_vol,
        time_range,
        price_vs_time,
        delta_vs_time,
        theta_vs_time,
    }
}

pub fn put_call_parity_residual(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    let norm = Normal::new(0.0, 1.0).unwrap();

    let call_p = Params {
        s, k, t, r, sigma,
        option_type: OptionType::Call,
    };
    let put_p = Params {
        s, k, t, r, sigma,
        option_type: OptionType::Put,
    };
    let c = price(&call_p, &norm);
    let p = price(&put_p, &norm);
    c - p - s + k * (-r * t).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call_params() -> Params {
        Params {
            s: 100.0,
            k: 100.0,
            t: 1.0,
            r: 0.05,
            sigma: 0.2,
            option_type: OptionType::Call,
        }
    }

    fn put_params() -> Params {
        Params {
            s: 100.0,
            k: 100.0,
            t: 1.0,
            r: 0.05,
            sigma: 0.2,
            option_type: OptionType::Put,
        }
    }

    #[test]
    fn test_norm_cdf_symmetry() {
        let norm = Normal::new(0.0, 1.0).unwrap();

        assert!((norm.cdf(0.0) - 0.5).abs() < 1e-10);
        assert!((norm.cdf(3.0) + norm.cdf(-3.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_atm_call_price() {
        let norm = Normal::new(0.0, 1.0).unwrap();

        // ATM call with S=K=100, T=1, r=5%, σ=20%
        // Analytical: ≈ 10.4506
        let p = call_params();
        let pr = price(&p, &norm);
        assert!((pr - 10.4506).abs() < 0.05, "Call price was {}", pr);
    }

    #[test]
    fn test_put_call_parity() {
        let residual = put_call_parity_residual(100.0, 100.0, 1.0, 0.05, 0.2);
        assert!(
            residual.abs() < 1e-10,
            "Put-call parity violated: residual = {}",
            residual
        );
    }

    #[test]
    fn test_delta_bounds() {
        let norm = Normal::new(0.0, 1.0).unwrap();

        let p = call_params();
        let d = delta(&p, &norm);
        assert!(d > 0.0 && d < 1.0, "Call delta out of [0,1]: {}", d);

        let p = put_params();
        let d = delta(&p, &norm);
        assert!(d > -1.0 && d < 0.0, "Put delta out of [-1,0]: {}", d);
    }

    #[test]
    fn test_gamma_positive() {
        let norm = Normal::new(0.0, 1.0).unwrap();

        let p = call_params();
        let g = gamma(&p, &norm);
        assert!(g > 0.0, "Gamma should be positive: {}", g);
    }

    #[test]
    fn test_vega_positive() {
        let norm = Normal::new(0.0, 1.0).unwrap();

        let p = call_params();
        let v = vega(&p, &norm);
        assert!(v > 0.0, "Vega should be positive: {}", v);
    }

    #[test]
    fn test_call_put_gamma_equal() {
        let norm = Normal::new(0.0, 1.0).unwrap();

        // Gamma is identical for call and put at the same strike
        let gc = gamma(&call_params(), &norm);
        let gp = gamma(&put_params(), &norm);
        assert!(
            (gc - gp).abs() < 1e-10,
            "Call gamma {} != Put gamma {}",
            gc,
            gp
        );
    }

    #[test]
    fn test_call_put_vega_equal() {
        let norm = Normal::new(0.0, 1.0).unwrap();

        // Vega is identical for call and put at the same strike
        let vc = vega(&call_params(), &norm);
        let vp = vega(&put_params(), &norm);
        assert!(
            (vc - vp).abs() < 1e-10,
            "Call vega {} != Put vega {}",
            vc,
            vp
        );
    }

    #[test]
    fn test_surface_generation() {
        let p = call_params();
        let surf = generate_surfaces(&p, 50);
        assert_eq!(surf.spot_range.len(), 50);
        assert_eq!(surf.prices.len(), 50);
        assert_eq!(surf.vol_range.len(), 50);
        assert_eq!(surf.time_range.len(), 50);
    }
}