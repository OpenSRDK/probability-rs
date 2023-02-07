use super::AcquisitionFunctions;
// use crate::{Normal, NormalParams};

pub struct ExpectedImprovement {
    f_vec: Vec<f64>,
}

impl AcquisitionFunctions for ExpectedImprovement {
    fn value(&self, theta: &crate::NormalParams) -> f64 {
        let mu = theta.mu();
        let sigma = theta.sigma();
        let tau = self.f_vec.iter().max().unwrap();
        let t = (mu - tau) / sigma;
        // let n = Normal;
        // let phi_large = n.p_kernel(n, t, &NormalParams::new(0.0, 1.0).unwrap());
        let phi_large = pdf(t);
        let phi_small = cdf(t);

        (mu - tau) * phi_large + sigma * phi_small
    }
}

// Abramowitz and Stegun (1964) formula 26.2.17
// precision: abs(err) < 7.5e-8

fn pdf(x: f64) -> f64 {
    ((-x * x) / 2.0).exp() / (2.0 * std::f64::consts::PI)
}
fn cdf(x: f64) -> f64 {
    // constants
    const p: f64 = 0.2316419;
    const b1: f64 = 0.31938153;
    const b2: f64 = -0.356563782;
    const b3: f64 = 1.781477937;
    const b4: f64 = -1.821255978;
    const b5: f64 = 1.330274429;

    let t = 1.0 / (1.0 + p * x.abs());
    let z = (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let y = 1.0 - z * ((((b5 * t + b4) * t + b3) * t + b2) * t + b1) * t;

    return if x > 0.0 { y } else { 1.0 - y };
}
