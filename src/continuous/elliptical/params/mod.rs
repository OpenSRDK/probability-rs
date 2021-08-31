pub mod exact;

use crate::{DistributionError, EllipticalError};
pub use exact::*;
use opensrdk_linear_algebra::*;
use rayon::prelude::*;
use std::fmt::Debug;

pub trait EllipticalParams: Clone + Debug + PartialEq {
    fn mu(&self) -> &Vec<f64>;

    fn x_mu(&self, x: &[f64]) -> Result<Vec<f64>, DistributionError> {
        let mu = self.mu();
        let n = mu.len();

        if n != x.len() {
            return Err(DistributionError::InvalidParameters(
                EllipticalError::DimensionMismatch.into(),
            ));
        }

        let x_mu = x
            .par_iter()
            .zip(mu.par_iter())
            .map(|(&xi, &mui)| xi - mui)
            .collect::<Vec<_>>()
            .col_mat();

        x_mu
    }

    fn sigma_inv_mul(&self, v: Vec<f64>) -> Result<Vec<f64>, DistributionError>;
    fn sigma_det_sqrt(&self) -> Result<f64, DistributionError>;

    fn lsigma_cols(&self) -> usize;
    fn sample(&self, z: Vec<f64>) -> Result<Vec<f64>, DistributionError>;
}
