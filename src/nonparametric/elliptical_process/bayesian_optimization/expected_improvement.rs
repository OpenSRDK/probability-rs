use super::AcquisitionFunctions;
use rand_distr::StandardNormal;

pub struct ExpectedImprovement;

impl AcquisitionFunctions for ExpectedImprovement {
    fn value(&self, theta: &crate::NormalParams, f_vec: Vec<f64>, xs: f64) -> f64 {
        let mu = theta.mu();
        let sigma = theta.sigma();
        let tau = f_vec.iter().max().unwrap();
        let f = todo!();
        let t = (mu - tau) / sigma;

        todo!()
    }
}
