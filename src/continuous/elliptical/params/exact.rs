use crate::{DistributionError, EllipticalError, EllipticalParams};
use opensrdk_linear_algebra::{matrix::ge::sy_he::po::trf::POTRF, *};

#[derive(Clone, Debug)]
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

    fn sigma_inv_mul(&self, v: Matrix) -> Result<Matrix, DistributionError> {
        Ok(POTRF(self.lsigma).potrs(v)?)
    }

    fn sigma_det_sqrt(&self) -> f64 {
        self.lsigma.trdet()
    }

    fn lsigma_cols(&self) -> usize {
        self.lsigma.cols()
    }

    fn sample(&self, z: Vec<f64>) -> Result<Vec<f64>, DistributionError> {
        Ok(self
            .mu
            .clone()
            .col_mat()
            .gemm(&self.lsigma, &z.col_mat(), 1.0, 1.0)?
            .vec())
    }
}
