use num_integer::sqrt;

use super::AcquisitionFunctions;

pub struct UpperConfidenceBound {
    pub trial: usize,
}

impl AcquisitionFunctions for UpperConfidenceBound {
    fn value(&self, theta: &crate::NormalParams) -> f64 {
        let mu = theta.mu();
        let sigma = theta.sigma();
        let n = self.trial as f64;
        let k = sqrt((n.ln() / n) as i64) as f64;

        mu + k * sigma
    }
}
