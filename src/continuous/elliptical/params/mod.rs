pub mod exact;

use crate::{DistributionError, EllipticalError};
pub use exact::*;
use opensrdk_linear_algebra::*;
use rayon::prelude::*;
use std::fmt::Debug;

pub trait EllipticalParams: Clone + Debug + PartialEq {
    fn n(&self) -> usize {
        self.mu().len()
    }
    fn mu(&self) -> &Vec<f64>;

    fn x_mu(&self, x: &[f64]) -> Result<Vec<f64>, DistributionError> {
        let n = x.len();

        if n != self.n() {
            return Err(DistributionError::InvalidParameters(
                EllipticalError::DimensionMismatch.into(),
            ));
        }

        let x_mu = x
            .par_iter()
            .zip(self.mu.par_iter())
            .map(|(&xi, &mui)| xi - mui)
            .collect::<Vec<_>>()
            .col_mat();

        x_mu
    }
    fn x_mu_t_sigma_inv_x_mu(&self, x_mu: Vec<f64>) -> Result<f64, DistributionError>;
    fn lsigma_det(&self) -> Result<f64, DistributionError>;

    fn z_len_for_sample(&self) -> usize;
    fn sample_from_z(&self, z: Vec<f64>) -> Result<Vec<f64>, DistributionError>;
}
