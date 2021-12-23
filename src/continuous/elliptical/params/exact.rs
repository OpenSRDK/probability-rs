use crate::TransformVec;
use crate::{DistributionError, EllipticalError, EllipticalParams};
use opensrdk_linear_algebra::{matrix::ge::sy_he::po::trf::POTRF, *};

#[derive(Clone, Debug)]
pub struct ExactEllipticalParams {
    mu: Vec<f64>,
    lsigma: POTRF,
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
        let lsigma = POTRF(lsigma);

        Ok(Self { mu, lsigma })
    }

    pub fn mu(&self) -> &Vec<f64> {
        &self.mu
    }

    pub fn lsigma(&self) -> &Matrix {
        &self.lsigma.0
    }

    pub fn eject(self) -> (Vec<f64>, Matrix) {
        (self.mu, self.lsigma.0)
    }
}

impl EllipticalParams for ExactEllipticalParams {
    fn mu(&self) -> &Vec<f64> {
        self.mu()
    }

    fn sigma_inv_mul(&self, v: Matrix) -> Result<Matrix, DistributionError> {
        Ok(self.lsigma.potrs(v)?)
    }

    fn lsigma_cols(&self) -> usize {
        self.lsigma.0.cols()
    }

    fn sample(&self, z: Vec<f64>) -> Result<Vec<f64>, DistributionError> {
        Ok(self
            .mu
            .clone()
            .col_mat()
            .gemm(&self.lsigma.0, &z.col_mat(), 1.0, 1.0)?
            .vec())
    }
}

impl TransformVec for ExactEllipticalParams {
    type T = usize;

    fn transform_vec(self) -> (Vec<f64>, Self::T) {
        let n = self.mu.len();
        ([self.mu, self.lsigma.0.vec()].concat(), n)
    }

    fn restore(v: Vec<f64>, info: Self::T) -> Self {
        let n = info;
        let mu = v[0..n].to_vec();
        let lsigma = Matrix::from(n, v[n..n + n * n].to_vec()).unwrap();
        Self::new(mu, lsigma).unwrap()
    }
}
