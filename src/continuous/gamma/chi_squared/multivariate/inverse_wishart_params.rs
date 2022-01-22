use opensrdk_linear_algebra::Matrix;

use crate::{DistributionError, InverseWishartError, RandomVariable};

#[derive(Clone, Debug, PartialEq)]
pub struct InverseWishartParams {
    lpsi: Matrix,
    nu: f64,
}

impl InverseWishartParams {
    pub fn new(lpsi: Matrix, nu: f64) -> Result<Self, DistributionError> {
        let p = lpsi.rows();
        if p != lpsi.cols() {
            return Err(DistributionError::InvalidParameters(
                InverseWishartError::DimensionMismatch.into(),
            ));
        }
        if nu <= p as f64 - 1.0 as f64 {
            return Err(DistributionError::InvalidParameters(
                InverseWishartError::NuMustBeGTEDimension.into(),
            ));
        }

        Ok(Self { lpsi, nu })
    }

    pub fn lpsi(&self) -> &Matrix {
        &self.lpsi
    }

    pub fn nu(&self) -> f64 {
        self.nu
    }
}

impl RandomVariable for InverseWishartParams {
    type RestoreInfo = usize;

    fn transform_vec(self) -> (Vec<f64>, Self::RestoreInfo) {
        todo!()
    }

    fn restore(v: Vec<f64>, info: Self::RestoreInfo) -> Self {
        todo!()
    }
}
