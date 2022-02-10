use crate::{DistributionError, EllipticalError, EllipticalParams, RandomVariable};
use opensrdk_linear_algebra::{pp::trf::PPTRF, *};

#[derive(Clone, Debug)]
pub struct ExactEllipticalParams {
    mu: Vec<f64>,
    lsigma: PPTRF,
}

impl ExactEllipticalParams {
    /// # Multivariate normal
    /// `L` is needed as second argument under decomposition `Sigma = L * L^T`
    /// l_sigma = sigma.pptrf()?;
    pub fn new(mu: Vec<f64>, lsigma: PPTRF) -> Result<Self, DistributionError> {
        let p = mu.len();
        if p != lsigma.0.dim() {
            return Err(DistributionError::InvalidParameters(
                EllipticalError::DimensionMismatch.into(),
            ));
        }

        Ok(Self { mu, lsigma })
    }

    pub fn mu(&self) -> &Vec<f64> {
        &self.mu
    }

    pub fn lsigma(&self) -> &PPTRF {
        &self.lsigma
    }

    pub fn eject(self) -> (Vec<f64>, PPTRF) {
        (self.mu, self.lsigma)
    }
}

impl RandomVariable for ExactEllipticalParams {
    type RestoreInfo = usize;

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        let n = self.mu.len();
        ([self.mu(), self.lsigma.0.elems()].concat(), n)
    }

    fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        if v.len() != info + info * (info + 1) / 2 {
            return Err(DistributionError::InvalidRestoreVector);
        }
        let n = *info;
        let mu = v[0..n].to_vec();
        let lsigma = PPTRF(SymmetricPackedMatrix::from(n, v[n..v.len()].to_vec()).unwrap());
        Self::new(mu, lsigma)
    }
}

impl EllipticalParams for ExactEllipticalParams {
    fn mu(&self) -> &Vec<f64> {
        self.mu()
    }

    fn sigma_inv_mul(&self, v: Matrix) -> Result<Matrix, DistributionError> {
        Ok(self.lsigma.pptrs(v)?)
    }

    fn lsigma_cols(&self) -> usize {
        self.lsigma.0.dim()
    }

    fn sample(&self, z: Vec<f64>) -> Result<Vec<f64>, DistributionError> {
        Ok(self
            .mu
            .clone()
            .col_mat()
            .gemm(&self.lsigma.0.to_mat(), &z.col_mat(), 1.0, 1.0)?
            .vec())
    }
}
