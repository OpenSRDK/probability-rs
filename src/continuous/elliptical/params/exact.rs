use crate::{DistributionError, EllipticalError, EllipticalParams};
use opensrdk_linear_algebra::*;
use rayon::prelude::*;
use std::fmt::Debug;

#[derive(Clone, Debug, PartialEq)]
pub struct ExactEllipticalParams {
    mu: Vec<f64>,
    lsigma: Matrix,
}

impl ExactEllipticalParams {
    /// # Multivariate normal
    /// `L` is needed as second argument under decomposition `Sigma = L * L^T`
    /// l_sigma = sigma.potrf()?;
    pub fn new(mu: Vec<f64>, lsigma: Matrix) -> Result<Self, DistributionError> {
        let p = mu.len();
        if p != lsigma.rows() || p != lsigma.cols() {
            return Err(DistributionError::InvalidParameters(
                EllipticalError::DimensionMismatch.into(),
            ));
        }

        Ok(Self { mu, lsigma })
    }

    pub fn mu(&self) -> &Vec<f64> {
        &self.mu
    }

    pub fn lsigma(&self) -> &Matrix {
        &self.lsigma
    }

    pub fn eject(self) -> (Vec<f64>, Matrix) {
        (self.mu, self.lsigma)
    }
}

impl EllipticalParams for ExactEllipticalParams {
    fn mu(&self) -> &Vec<f64> {
        self.mu()
    }

    fn x_mu_t_sigma_inv_x_mu(&self, x_mu: Vec<f64>) -> Result<f64, DistributionError> {
        Ok((x_mu.t() * self.lsigma.potrs(x_mu)?)[0][0])
    }

    fn lsigma_det(&self) -> Result<f64, DistributionError> {
        Ok(self.lsigma.trdet()?)
    }

    fn z_len_for_sample(&self) -> usize {
        self.lsigma.cols()
    }

    fn sample_from_z(&self, z: Vec<f64>) -> Result<Vec<f64>, DistributionError> {
        Ok(self
            .mu
            .clone()
            .col_mat()
            .gemm(&self.lsigma, &z.col_mat(), 1.0, 1.0)?)
    }
}
