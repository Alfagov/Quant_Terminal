use rand::Rng;
use serde::{Deserialize, Serialize};
use validator::Validate;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum Action {
    Buy,
    Sell,
}

const MAX_STEPS: usize = 10_000;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Validate)]
pub struct Params {
    #[validate(range(min = 0.0, max = 10000.0))]
    pub v_h: f64,
    #[validate(range(min = 0.0, max = 10000.0))]
    pub v_l: f64,
    #[validate(range(min = 0.0, max = 1.0))]
    pub prior_h: f64,
    #[validate(range(min = 0.0, max = 1.0))]
    pub alpha: f64,
    #[validate(range(min = 0.5, max = 1.0))]
    pub accuracy_informed: f64,
    #[validate(range(min = 1, max = 10000))]
    pub steps: usize,
    pub seed: u64,
    pub true_state_is_high: bool,
}

impl Params {
    pub fn validate_limits(&self) -> Result<(), String> {
        if self.steps > MAX_STEPS {
            return Err(format!("Steps cannot exceed {}", MAX_STEPS));
        }

        if self.v_l >= self.v_h {
            return Err("v_l must be greater than v_h".to_string());
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RoundData {
    pub t: usize,
    pub bid: f64,
    pub ask: f64,
    pub mid: f64,
    pub spread: f64,
    pub belief_p_h: f64,
    pub action: Action,
    pub informed: bool,
    pub true_state: bool,
    pub true_value: f64,
    pub transaction_price: f64,
    pub trade_pnl: f64,
    pub price_impact: f64,
}

pub struct Simulation {
    rng: ChaCha8Rng,
}

impl Simulation {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    fn posterior_buy(p: f64, alpha: f64, rho: f64) -> f64 {
        let like_h = alpha * rho + (1.0 - alpha) * 0.5;
        let like_l = alpha * (1.0 - rho) + (1.0 - alpha) * 0.5;
        let numerator = p * like_h;
        let denominator = numerator + (1.0 - p) * like_l;
        numerator / denominator
    }

    fn posterior_sell(p: f64, alpha: f64, rho: f64) -> f64 {
        let like_h = alpha * (1.0 - rho) + (1.0 - alpha) * 0.5;
        let like_l = alpha * rho + (1.0 - alpha) * 0.5;
        let numerator = p * like_h;
        let denominator = numerator + (1.0 - p) * like_l;
        numerator / denominator
    }

    fn quotes_from_prior(
        p: f64,
        v_h: f64,
        v_l: f64,
        alpha: f64,
        rho: f64,
    ) -> (f64, f64, f64, f64) {
        let p_buy = Self::posterior_buy(p, alpha, rho);
        let p_sell = Self::posterior_sell(p, alpha, rho);
        let ask = p_buy * v_h + (1.0 - p_buy) * v_l;
        let bid = p_sell * v_h + (1.0 - p_sell) * v_l;
        (bid, ask, p_sell, p_buy)
    }

    pub fn run(&mut self, params: &Params) -> Vec<RoundData> {
        let true_h = params.true_state_is_high;
        let true_value = if true_h {params.v_h} else {params.v_l};

        let mut rounds = Vec::with_capacity(params.steps);
        let mut p = params.prior_h;
        let mut prev_mid = 0.5 * (params.v_h + params.v_l);

        for t in 1..=params.steps {
            let (bid, ask, post_sell, post_buy) = Self::quotes_from_prior(p, params.v_h, params.v_l, params.alpha, params.accuracy_informed);

            let informed = self.rng.random_bool(params.alpha);
            let action = if informed {
                if true_h {
                    if self.rng.random_bool(params.accuracy_informed) {
                        Action::Buy
                    } else {
                        Action::Sell
                    }
                } else if self.rng.random_bool(params.accuracy_informed) {
                    Action::Sell
                } else {
                    Action::Buy
                }
            } else if self.rng.random_bool(0.5) {
                Action::Buy
            } else {
                Action::Sell
            };

            let (new_p, px) = match action {
                Action::Buy => (post_buy, ask),
                Action::Sell => (post_sell, bid),
            };

            let new_mid = 0.5 * (bid + ask);
            let trade_pnl = match action {
                Action::Buy => true_value - px,
                Action::Sell => px - true_value,
            };

            let price_impact = new_mid - prev_mid;

            rounds.push(RoundData {
                t,
                bid,
                ask,
                mid: new_mid,
                spread: ask - bid,
                belief_p_h: new_p,
                action,
                informed,
                true_state: true_h,
                true_value,
                transaction_price: px,
                trade_pnl,
                price_impact,
            });

            p = new_p;
            prev_mid = new_mid;
        }

        rounds
    }
}

pub fn cumulative_pnl(rounds: &[RoundData]) -> (Vec<f64>, Vec<f64>) {
    let mut informed_pnl = vec![0.0];
    let mut uninformed_pnl = vec![0.0];
    let mut cum_informed = 0.0;
    let mut cum_uninformed = 0.0;

    for round in rounds {
        if round.informed {
            cum_informed += round.trade_pnl;
            informed_pnl.push(cum_informed);
            uninformed_pnl.push(cum_uninformed);
        } else {
            cum_uninformed += round.trade_pnl;
            informed_pnl.push(cum_informed);
            uninformed_pnl.push(cum_uninformed);
        }
    }

    (informed_pnl, uninformed_pnl)
}

pub fn get_price_impacts(rounds: &[RoundData]) -> Vec<f64> {
    rounds.iter().map(|r| r.price_impact).collect()
}

pub fn calculate_returns(rounds: &[RoundData]) -> Vec<f64> {
    rounds
        .windows(2)
        .map(|w| (w[1].transaction_price / w[0].transaction_price).ln())
        .collect()
}

pub fn separate_trades(rounds: &[RoundData]) -> (Vec<&RoundData>, Vec<&RoundData>) {
    rounds.iter().partition(|r| r.informed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_posterior_buy_increases_belief() {
        let p_prior = 0.5;
        let alpha = 0.3;
        let rho = 0.7;
        let p_posterior = Simulation::posterior_buy(p_prior, alpha, rho);
        assert!(p_posterior > p_prior);
    }

    #[test]
    fn test_posterior_sell_decreases_belief() {
        let p_prior = 0.5;
        let alpha = 0.3;
        let rho = 0.7;
        let p_posterior = Simulation::posterior_sell(p_prior, alpha, rho);
        assert!(p_posterior < p_prior);
    }

    #[test]
    fn test_simulation_produces_correct_rounds() {
        let params = Params {
            v_h: 100.0,
            v_l: 50.0,
            prior_h: 0.5,
            alpha: 0.3,
            accuracy_informed: 0.7,
            steps: 10,
            seed: 42,
            true_state_is_high: false,
        };
        let mut sim = Simulation::new(params.seed);
        let rounds = sim.run(&params);
        assert_eq!(rounds.len(), 10);
    }

    #[test]
    fn test_bid_less_than_ask() {
        let (bid, ask, _, _) = Simulation::quotes_from_prior(0.5, 100.0, 50.0, 0.3, 0.7);
        assert!(bid <= ask);
    }
}