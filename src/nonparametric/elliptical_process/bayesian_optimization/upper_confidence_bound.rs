use num_integer::sqrt;

use super::AcquisitionFunctions;

pub struct UpperConfidenceBound {
    trial: f64,
}

impl AcquisitionFunctions for UpperConfidenceBound {
    fn value(&self, theta: &crate::NormalParams) -> f64 {
        let mu = theta.mu();
        let sigma = theta.sigma();
        let n = self.trial;
        let k = sqrt(n.ln() / n);

        mu + k * sigma
    }
}
