use crate::{DistributionError, RandomVariable, WishartError};
use opensrdk_linear_algebra::Matrix;

#[derive(Clone, Debug, PartialEq)]
pub struct WishartParams {
    lv: Matrix,
    n: f64,
}

impl WishartParams {
    pub fn new(lv: Matrix, n: f64) -> Result<Self, DistributionError> {
        let p = lv.rows();
        if p != lv.cols() {
            return Err(DistributionError::InvalidParameters(
                WishartError::DimensionMismatch.into(),
            ));
        }
        if n <= p as f64 - 1.0 as f64 {
            return Err(DistributionError::InvalidParameters(
                WishartError::NMustBeGTEDimension.into(),
            ));
        }

        Ok(Self { lv, n })
    }

    pub fn lv(&self) -> &Matrix {
        &self.lv
    }

    pub fn n(&self) -> f64 {
        self.n
    }
}

impl RandomVariable for WishartParams {
    type RestoreInfo = ();

    fn transform_vec(self) -> (Vec<f64>, Self::RestoreInfo) {
        todo!()
    }

    fn restore(v: Vec<f64>, info: Self::RestoreInfo) -> Self {
        todo!()
    }
}
