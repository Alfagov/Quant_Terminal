use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bond {
    pub face_value: f64,
    pub ytm: f64,               // Annualized Yield to Maturity (decimal, e.g., 0.05 for 5%)
    pub maturity: f64,          // Years to maturity
    pub coupon_yield: f64,      // Annualized Coupon Rate (decimal)
    pub coupon_interval: f64,   // Time between coupons in years (e.g., 0.5 for semiannual)
}

/// Single-bond analysis results
pub struct BondAnalysis {
    pub price: f64,
    pub macaulay_duration: f64,
    pub modified_duration: f64,
    pub convexity: f64,
    pub yield_shifts_bps: Vec<f64>,
    pub real_price_change: Vec<f64>,
    pub duration_approx: Vec<f64>,
    pub convexity_approx: Vec<f64>,
    pub absolute_prices: Vec<f64>,
}

/// Portfolio-level analysis results
pub struct PortfolioAnalysis {
    pub total_price: f64,
    pub weighted_mod_duration: f64,
    pub weighted_mac_duration: f64,
    pub weighted_convexity: f64,
    pub yield_shifts_bps: Vec<f64>,
    pub real_price_change: Vec<f64>,
    pub duration_approx: Vec<f64>,
    pub convexity_approx: Vec<f64>,
    pub duration_error: Vec<f64>,
    pub convexity_error: Vec<f64>,
    pub bond_analyses: Vec<BondAnalysis>,
}

impl Bond {
    pub fn periods(&self) -> u64 {
        (self.maturity / self.coupon_interval).round() as u64
    }

    pub fn price(&self) -> f64 {
        let periods = self.periods();
        let periodic_rate = self.ytm * self.coupon_interval;
        let periodic_coupon = self.face_value * self.coupon_yield * self.coupon_interval;

        let mut pv_coupons = 0.0;

        for i in 1..= periods {
            pv_coupons += periodic_coupon / (1.0 + periodic_rate).powi(i as i32);
        }

        let pv_face = self.face_value / (1.0 + periodic_rate).powi(periods as i32);

        pv_coupons + pv_face
    }

    pub fn price_at_ytm(&self, ytm: f64) -> f64 {
        let periods = self.periods();
        let periodic_rate = ytm * self.coupon_interval;
        let periodic_coupon = self.face_value * self.coupon_yield * self.coupon_interval;

        let mut pv_coupons = 0.0;
        for i in 1..=periods {
            pv_coupons += periodic_coupon / (1.0 + periodic_rate).powi(i as i32);
        }

        let pv_face = self.face_value / (1.0 + periodic_rate).powi(periods as i32);
        pv_coupons + pv_face
    }

    pub fn macaulay_duration(&self) -> f64 {
        let price = self.price();
        if price == 0.0 { return 0.0; }

        let periods = self.periods();
        let periodic_rate = self.ytm * self.coupon_interval;
        let periodic_coupon = self.face_value * self.coupon_yield * self.coupon_interval;

        let mut weighted_time_sum = 0.0;

        for i in 1..=periods {
            let time_in_years = i as f64 * self.coupon_interval;
            let pv_flow = periodic_coupon / (1.0 + periodic_rate).powi(i as i32);
            weighted_time_sum += pv_flow * time_in_years;
        }

        let time_at_maturity = periods as f64 * self.coupon_interval;
        let pv_face = self.face_value / (1.0 + periodic_rate).powi(periods as i32);
        weighted_time_sum += pv_face * time_at_maturity;

        weighted_time_sum / price
    }

    pub fn modified_duration(&self) -> f64 {
        let mac_duration = self.macaulay_duration();
        let periodic_rate = self.ytm * self.coupon_interval;

        mac_duration / (1.0 + periodic_rate)
    }

    pub fn convexity(&self) -> f64 {
        let price = self.price();
        if price == 0.0 { return 0.0; }

        let periods = self.periods();
        let periodic_rate = self.ytm * self.coupon_interval;
        let periodic_coupon = self.face_value * self.coupon_yield * self.coupon_interval;

        let interval_sq = self.coupon_interval.powi(2);
        let mut weighted_convexity_sum = 0.0;

        for i in 1..=periods {
            let t = i as f64; // Period number
            let pv_flow = periodic_coupon / (1.0 + periodic_rate).powi(i as i32);

            // The term t * (t + 1) captures the second derivative logic
            weighted_convexity_sum += pv_flow * (t * (t + 1.0));
        }

        let t_final = periods as f64;
        let pv_face = self.face_value / (1.0 + periodic_rate).powi(periods as i32);
        weighted_convexity_sum += pv_face * (t_final * (t_final + 1.0));

        (weighted_convexity_sum * interval_sq) / (price * (1.0 + periodic_rate).powi(2))
    }
}

/// Analyze a single bond across yield shifts from -300bps to +300bps
pub fn analyze_bond(bond: &Bond, num_points: usize) -> BondAnalysis {
    let price = bond.price();
    let mac_dur = bond.macaulay_duration();
    let mod_dur = bond.modified_duration();
    let conv = bond.convexity();

    let min_bps = -300.0_f64;
    let max_bps = 300.0_f64;
    let step = (max_bps - min_bps) / (num_points - 1) as f64;

    let mut yield_shifts_bps = Vec::with_capacity(num_points);
    let mut real_price_change = Vec::with_capacity(num_points);
    let mut duration_approx = Vec::with_capacity(num_points);
    let mut convexity_approx = Vec::with_capacity(num_points);
    let mut absolute_prices = Vec::with_capacity(num_points);

    for i in 0..num_points {
        let bps = min_bps + step * i as f64;
        let dy = bps / 10000.0; // convert bps to decimal
        let shifted_ytm = bond.ytm + dy;

        let new_price = bond.price_at_ytm(shifted_ytm);
        let real_change = new_price - price;
        let dur_approx = -mod_dur * dy * price;
        let conv_approx = -mod_dur * dy * price + 0.5 * conv * dy * dy * price;

        yield_shifts_bps.push(bps);
        real_price_change.push(real_change);
        duration_approx.push(dur_approx);
        convexity_approx.push(conv_approx);
        absolute_prices.push(new_price);
    }

    BondAnalysis {
        price,
        macaulay_duration: mac_dur,
        modified_duration: mod_dur,
        convexity: conv,
        yield_shifts_bps,
        real_price_change,
        duration_approx,
        convexity_approx,
        absolute_prices,
    }
}

/// Analyze a portfolio of bonds
pub fn analyze_portfolio(bonds: &[Bond], num_points: usize) -> PortfolioAnalysis {
    let bond_analyses: Vec<BondAnalysis> = bonds.iter()
        .map(|b| analyze_bond(b, num_points))
        .collect();

    let total_price: f64 = bond_analyses.iter().map(|a| a.price).sum();

    // Market-value-weighted averages
    let weighted_mod_duration = if total_price > 0.0 {
        bond_analyses.iter()
            .map(|a| a.modified_duration * a.price)
            .sum::<f64>() / total_price
    } else {
        0.0
    };

    let weighted_mac_duration = if total_price > 0.0 {
        bond_analyses.iter()
            .map(|a| a.macaulay_duration * a.price)
            .sum::<f64>() / total_price
    } else {
        0.0
    };

    let weighted_convexity = if total_price > 0.0 {
        bond_analyses.iter()
            .map(|a| a.convexity * a.price)
            .sum::<f64>() / total_price
    } else {
        0.0
    };

    // Portfolio-level curves using weighted metrics
    let min_bps = -300.0_f64;
    let max_bps = 300.0_f64;
    let step = (max_bps - min_bps) / (num_points - 1) as f64;

    let mut yield_shifts_bps = Vec::with_capacity(num_points);
    let mut real_price_change = Vec::with_capacity(num_points);
    let mut duration_approx = Vec::with_capacity(num_points);
    let mut convexity_approx = Vec::with_capacity(num_points);
    let mut duration_error = Vec::with_capacity(num_points);
    let mut convexity_error = Vec::with_capacity(num_points);

    for i in 0..num_points {
        let bps = min_bps + step * i as f64;
        let dy = bps / 10000.0;

        // Real portfolio price change = sum of individual repriced bonds - original total
        let portfolio_new_price: f64 = bonds.iter()
            .map(|b| b.price_at_ytm(b.ytm + dy))
            .sum();
        let real_change = portfolio_new_price - total_price;

        let dur_approx = -weighted_mod_duration * dy * total_price;
        let conv_approx = -weighted_mod_duration * dy * total_price
            + 0.5 * weighted_convexity * dy * dy * total_price;

        yield_shifts_bps.push(bps);
        real_price_change.push(real_change);
        duration_approx.push(dur_approx);
        convexity_approx.push(conv_approx);
        duration_error.push(real_change - dur_approx);
        convexity_error.push(real_change - conv_approx);
    }

    PortfolioAnalysis {
        total_price,
        weighted_mod_duration,
        weighted_mac_duration,
        weighted_convexity,
        yield_shifts_bps,
        real_price_change,
        duration_approx,
        convexity_approx,
        duration_error,
        convexity_error,
        bond_analyses,
    }
}