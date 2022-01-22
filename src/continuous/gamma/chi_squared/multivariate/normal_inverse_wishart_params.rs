use crate::{DistributionError, NormalInverseWishartError, RandomVariable};
use opensrdk_linear_algebra::Matrix;

#[derive(Clone, Debug, PartialEq)]
pub struct NormalInverseWishartParams {
    mu0: Vec<f64>,
    lambda: f64,
    lpsi: Matrix,
    nu: f64,
}

impl NormalInverseWishartParams {
    pub fn new(
        mu0: Vec<f64>,
        lambda: f64,
        lpsi: Matrix,
        nu: f64,
    ) -> Result<Self, DistributionError> {
        let n = mu0.len();
        if n != lpsi.rows() || n != lpsi.cols() {
            return Err(DistributionError::InvalidParameters(
                NormalInverseWishartError::DimensionMismatch.into(),
            ));
        }
        if lambda <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                NormalInverseWishartError::DimensionMismatch.into(),
            ));
        }
        if nu <= n as f64 - 1.0 {
            return Err(DistributionError::InvalidParameters(
                NormalInverseWishartError::DimensionMismatch.into(),
            ));
        }

        Ok(Self {
            mu0,
            lambda,
            lpsi,
            nu,
        })
    }

    pub fn mu0(&self) -> &Vec<f64> {
        &self.mu0
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    pub fn lpsi(&self) -> &Matrix {
        &self.lpsi
    }

    pub fn nu(&self) -> f64 {
        self.nu
    }
}

impl RandomVariable for NormalInverseWishartParams {
    type RestoreInfo = usize;

    fn transform_vec(self) -> (Vec<f64>, Self::RestoreInfo) {
        todo!()
    }

    fn restore(v: Vec<f64>, info: Self::RestoreInfo) -> Self {
        todo!()
    }
}
