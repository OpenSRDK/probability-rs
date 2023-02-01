use super::AcquisitionFunctions;
use crate::{Normal, NormalParams};

pub struct ExpectedImprovement {
    f_vec: Vec<f64>,
}

impl AcquisitionFunctions for ExpectedImprovement {
    fn value(&self, theta: &crate::NormalParams) -> f64 {
        let mu = theta.mu();
        let sigma = theta.sigma();
        let tau = self.f_vec.iter().max().unwrap();
        let t = (mu - tau) / sigma;
        let n = Normal;
        let phi_large = n.p_kernel(n, t, &NormalParams::new(0.0, 1.0).unwrap());
        (mu - tau) + sigma
    }
}
