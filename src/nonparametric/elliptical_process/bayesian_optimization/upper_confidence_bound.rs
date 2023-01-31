use num_integer::sqrt;

use super::AcquisitionFunctions;

pub struct UpperConfidenceBound;

impl AcquisitionFunctions for UpperConfidenceBound {
    fn value(&self, theta: &crate::NormalParams, n: usize) -> f64 {
        let mu = theta.mu();
        let sigma = theta.sigma();
        let n = n as f64;
        let k = sqrt((n.ln() / n));

        mu + k * sigma
    }
}
