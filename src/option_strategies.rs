use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::distribution::Normal;
use crate::black_scholes;
use crate::black_scholes::{Greeks, OptionType};
use crate::utils::unzip8;

const MAX_STRATEGY_POINTS: usize = 200;
const MAX_LEGS: usize = 10;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Leg {
    pub option_type: OptionType,
    pub strike: f64,
    pub quantity: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyParams {
    pub spot: f64,
    pub t: f64,
    pub r: f64,
    pub sigma: f64,
    pub legs: Vec<Leg>,
    pub num_points: usize,
}

impl StrategyParams {
    pub fn validate(&self) -> Result<(), String> {
        if self.spot <= 0.0 {
            return Err("Spot price must be positive".into());
        }
        if self.t <= 0.0 {
            return Err("Time to expiry must be positive".into());
        }
        if self.sigma <= 0.0 {
            return Err("Volatility must be positive".into());
        }
        if self.legs.is_empty() {
            return Err("Strategy must have at least one leg".into());
        }
        if self.legs.len() > MAX_LEGS {
            return Err(format!("Maximum {} legs allowed", MAX_LEGS));
        }
        for (i, leg) in self.legs.iter().enumerate() {
            if leg.strike <= 0.0 {
                return Err(format!("Leg {} strike must be positive", i + 1));
            }
            if leg.quantity == 0 {
                return Err(format!("Leg {} quantity cannot be zero", i + 1));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct StrategyResult {
    pub leg_prices: Vec<f64>,
    pub leg_greeks: Vec<Greeks>,

    pub net_premium: f64,
    pub net_greeks: Greeks,
    pub max_profit: f64,
    pub max_loss: f64,
    pub breakeven_points: Vec<f64>,

    pub spot_range: Vec<f64>,
    pub payoff_at_expiry: Vec<f64>,
    pub pnl_at_expiry: Vec<f64>,
    pub current_value: Vec<f64>,

    pub net_deltas: Vec<f64>,
    pub net_gammas: Vec<f64>,
    pub net_thetas: Vec<f64>,
    pub net_vegas: Vec<f64>,
    pub net_rhos: Vec<f64>,
}

fn leg_payoff_at_expiry(leg: &Leg, spot: f64) -> f64 {
    let intrinsic = match leg.option_type {
        OptionType::Call => (spot - leg.strike).max(0.0),
        OptionType::Put => (leg.strike - spot).max(0.0),
    };
    leg.quantity as f64 * intrinsic
}

fn strategy_payoff_at_expiry(legs: &[Leg], spot: f64) -> f64 {
    legs.iter().map(|leg| leg_payoff_at_expiry(leg, spot)).sum()
}

fn aggregate_greeks(legs: &[Leg], leg_greeks: &[Greeks]) -> Greeks {
    let mut agg = Greeks {
        delta: 0.0,
        gamma: 0.0,
        theta: 0.0,
        vega: 0.0,
        rho: 0.0,
    };
    for (leg, gs) in legs.iter().zip(leg_greeks.iter()) {
        let q = leg.quantity as f64;
        agg.delta += q * gs.delta;
        agg.gamma += q * gs.gamma;
        agg.theta += q * gs.theta;
        agg.vega += q * gs.vega;
        agg.rho += q * gs.rho;
    }
    agg
}

fn find_zero_crossings(xs: &[f64], ys: &[f64]) -> Vec<f64> {
    let mut crossings = Vec::new();
    for i in 0..ys.len() - 1 {
        if ys[i].signum() != ys[i + 1].signum() && ys[i] != 0.0 && ys[i + 1] != 0.0 {
            let x = xs[i] + (xs[i + 1] - xs[i]) * (-ys[i] / (ys[i + 1] - ys[i]));
            crossings.push(x);
        }
    }
    crossings
}

pub fn analyze_strategy(params: &StrategyParams) -> StrategyResult {
    let norm = Normal::new(0.0, 1.0).unwrap();
    let n = params.num_points.min(MAX_STRATEGY_POINTS);

    let min_strike = params
        .legs
        .iter()
        .map(|l| l.strike)
        .fold(f64::INFINITY, f64::min);

    let max_strike = params
        .legs
        .iter()
        .map(|l| l.strike)
        .fold(f64::NEG_INFINITY, f64::max);

    let s_min = (min_strike * 0.5).max(0.01);
    let s_max = max_strike * 1.5;
    let ds = (s_max - s_min) / (n - 1) as f64;
    let spot_range: Vec<f64> = (0..n).map(|i| s_min + i as f64 * ds).collect();

    let leg_prices: Vec<f64> = params
        .legs
        .iter()
        .map(|leg| {
            let bp = black_scholes::Params {
                s: params.spot,
                k: leg.strike,
                t: params.t,
                r: params.r,
                sigma: params.sigma,
                option_type: leg.option_type,
            };
            black_scholes::price(&bp, &norm)
        })
        .collect();

    let leg_greeks: Vec<Greeks> = params
        .legs
        .iter()
        .map(|leg| {
            let bp = black_scholes::Params {
                s: params.spot,
                k: leg.strike,
                t: params.t,
                r: params.r,
                sigma: params.sigma,
                option_type: leg.option_type,
            };
            black_scholes::greeks(&bp, &norm)
        })
        .collect();

    let net_premium: f64 = params
        .legs
        .iter()
        .zip(leg_prices.iter())
        .map(|(leg, &price)| leg.quantity as f64 * price)
        .sum();

    let net_greeks = aggregate_greeks(&params.legs, &leg_greeks);

    let legs_clone = params.legs.clone();
    let curve_data: Vec<(f64, f64, f64, f64, f64, f64, f64, f64)> = spot_range
        .par_iter()
        .map(|&s| {
            let norm_local = Normal::new(0.0, 1.0).unwrap();
            let payoff = strategy_payoff_at_expiry(&legs_clone, s);
            let pnl = payoff - net_premium;

            let (mut cv, mut d, mut g, mut th, mut v, mut rh) =
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            for leg in &legs_clone {
                let bp = black_scholes::Params {
                    s,
                    k: leg.strike,
                    t: params.t,
                    r: params.r,
                    sigma: params.sigma,
                    option_type: leg.option_type,
                };
                let q = leg.quantity as f64;
                cv += q * black_scholes::price(&bp, &norm_local);
                let gs = black_scholes::greeks(&bp, &norm_local);
                d += q * gs.delta;
                g += q * gs.gamma;
                th += q * gs.theta / 365.0;
                v += q * gs.vega / 100.0;
                rh += q * gs.rho / 100.0;
            }
            (payoff, pnl, cv, d, g, th, v, rh)
        })
        .collect();

    let (payoff_at_expiry, pnl_at_expiry, current_value, net_deltas, net_gammas, net_thetas, net_vegas, net_rhos) =
        unzip8(curve_data);

    let breakeven_points = find_zero_crossings(&spot_range, &pnl_at_expiry);

    let max_profit = pnl_at_expiry
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let max_loss = pnl_at_expiry
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);

    StrategyResult {
        leg_prices,
        leg_greeks,
        net_premium,
        net_greeks,
        max_profit,
        max_loss,
        breakeven_points,
        spot_range,
        payoff_at_expiry,
        pnl_at_expiry,
        current_value,
        net_deltas,
        net_gammas,
        net_thetas,
        net_vegas,
        net_rhos,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bull_call_spread() {
        let params = StrategyParams {
            spot: 100.0,
            t: 0.5,
            r: 0.05,
            sigma: 0.20,
            legs: vec![
                Leg {
                    option_type: OptionType::Call,
                    strike: 100.0,
                    quantity: 1,
                },
                Leg {
                    option_type: OptionType::Call,
                    strike: 110.0,
                    quantity: -1,
                },
            ],
            num_points: 100,
        };
        let result = analyze_strategy(&params);

        // Net premium should be positive (debit spread)
        assert!(result.net_premium > 0.0, "Bull call spread should be a debit");

        // Max loss approximately equals net premium
        assert!(
            (result.max_loss.abs() - result.net_premium).abs() < 0.5,
            "Max loss {} should approximate net premium {}",
            result.max_loss,
            result.net_premium
        );

        // Max profit approximately equals spread width minus premium
        let spread_width = 10.0;
        assert!(
            (result.max_profit - (spread_width - result.net_premium)).abs() < 0.5,
            "Max profit {} should approximate spread_width - premium {}",
            result.max_profit,
            spread_width - result.net_premium
        );
    }

    #[test]
    fn test_straddle_breakevens() {
        let params = StrategyParams {
            spot: 100.0,
            t: 0.5,
            r: 0.05,
            sigma: 0.20,
            legs: vec![
                Leg {
                    option_type: OptionType::Call,
                    strike: 100.0,
                    quantity: 1,
                },
                Leg {
                    option_type: OptionType::Put,
                    strike: 100.0,
                    quantity: 1,
                },
            ],
            num_points: 200,
        };
        let result = analyze_strategy(&params);

        assert_eq!(
            result.breakeven_points.len(),
            2,
            "Straddle should have 2 breakeven points, got {:?}",
            result.breakeven_points
        );

        let lower = result.breakeven_points[0];
        let upper = result.breakeven_points[1];
        assert!(lower < 100.0, "Lower breakeven {} should be below strike", lower);
        assert!(upper > 100.0, "Upper breakeven {} should be above strike", upper);
    }

    #[test]
    fn test_iron_condor() {
        let params = StrategyParams {
            spot: 100.0,
            t: 0.5,
            r: 0.05,
            sigma: 0.20,
            legs: vec![
                Leg {
                    option_type: OptionType::Put,
                    strike: 85.0,
                    quantity: 1,
                },
                Leg {
                    option_type: OptionType::Put,
                    strike: 95.0,
                    quantity: -1,
                },
                Leg {
                    option_type: OptionType::Call,
                    strike: 105.0,
                    quantity: -1,
                },
                Leg {
                    option_type: OptionType::Call,
                    strike: 115.0,
                    quantity: 1,
                },
            ],
            num_points: 200,
        };
        let result = analyze_strategy(&params);

        // Iron condor is a credit strategy
        assert!(
            result.net_premium < 0.0,
            "Iron condor should be a credit, got {}",
            result.net_premium
        );

        // Both max profit and max loss should be finite
        assert!(result.max_profit.is_finite());
        assert!(result.max_loss.is_finite());
    }

    #[test]
    fn test_validation() {
        let params = StrategyParams {
            spot: 100.0,
            t: 0.5,
            r: 0.05,
            sigma: 0.20,
            legs: vec![],
            num_points: 100,
        };
        assert!(params.validate().is_err());

        let params = StrategyParams {
            spot: -1.0,
            t: 0.5,
            r: 0.05,
            sigma: 0.20,
            legs: vec![Leg {
                option_type: OptionType::Call,
                strike: 100.0,
                quantity: 1,
            }],
            num_points: 100,
        };
        assert!(params.validate().is_err());

        let params = StrategyParams {
            spot: 100.0,
            t: 0.5,
            r: 0.05,
            sigma: 0.20,
            legs: vec![Leg {
                option_type: OptionType::Call,
                strike: 100.0,
                quantity: 0,
            }],
            num_points: 100,
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_single_long_call() {
        let params = StrategyParams {
            spot: 100.0,
            t: 1.0,
            r: 0.05,
            sigma: 0.20,
            legs: vec![Leg {
                option_type: OptionType::Call,
                strike: 100.0,
                quantity: 1,
            }],
            num_points: 100,
        };
        let result = analyze_strategy(&params);

        // Net premium should match BS price for ATM call
        let norm = Normal::new(0.0, 1.0).unwrap();
        let bs_price = black_scholes::price(
            &black_scholes::Params {
                s: 100.0,
                k: 100.0,
                t: 1.0,
                r: 0.05,
                sigma: 0.20,
                option_type: OptionType::Call,
            },
            &norm,
        );
        assert!(
            (result.net_premium - bs_price).abs() < 1e-8,
            "Single long call premium {} should equal BS price {}",
            result.net_premium,
            bs_price
        );

        // Max loss should be approximately the premium
        assert!(
            (result.max_loss.abs() - result.net_premium).abs() < 0.5,
            "Max loss should approximate premium paid"
        );
    }
}