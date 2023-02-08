use crate::{DistributionError, EllipticalError, EllipticalParams, RandomVariable};
use opensrdk_linear_algebra::{pp::trf::PPTRF, *};

#[derive(Clone, Debug)]
pub struct ExactEllipticalParams {
    pub mu: Vec<f64>,
    pub lsigma: PPTRF,
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

    fn len(&self) -> usize {
        let t = self.lsigma.0.elems().len();
        t + self.mu.len()
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
        // let lsigma = self.lsigma.0.to_mat();
        // let sigma_orig = lsigma.clone() * lsigma.t();
        // let sigma = PPTRF(SymmetricPackedMatrix::from_mat(&sigma_orig).unwrap());
        // Ok(sigma.pptrs(v)?)
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

#[cfg(test)]
mod tests {
    use crate::{EllipticalParams, ExactMultivariateNormalParams};
    use opensrdk_linear_algebra::{pp::trf::PPTRF, *};

    #[test]
    fn it_works() {
        let mu = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let lsigma = SymmetricPackedMatrix::from_mat(&mat!(
           1.0,  0.0,  0.0,  0.0,  0.0,  0.0;
           2.0,  3.0,  0.0,  0.0,  0.0,  0.0;
           4.0,  5.0,  6.0,  0.0,  0.0,  0.0;
           7.0,  8.0,  9.0, 10.0,  0.0,  0.0;
          11.0, 12.0, 13.0, 14.0, 15.0,  0.0;
          16.0, 17.0, 18.0, 19.0, 20.0, 21.0
        ))
        .unwrap();
        println!("{:#?}", lsigma);

        let theta = &ExactMultivariateNormalParams::new(mu, PPTRF(lsigma)).unwrap();
        let result = theta.lsigma.clone();

        println!("{:#?}", result);
    }
}
