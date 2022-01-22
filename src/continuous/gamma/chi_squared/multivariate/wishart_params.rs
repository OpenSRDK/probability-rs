use crate::{DistributionError, RandomVariable, WishartError};
use opensrdk_linear_algebra::pp::trf::PPTRF;

#[derive(Clone, Debug, PartialEq)]
pub struct WishartParams {
    lv: PPTRF,
    n: f64,
}

impl WishartParams {
    pub fn new(lv: PPTRF, n: f64) -> Result<Self, DistributionError> {
        let p = lv.0.dim();
        if n <= p as f64 - 1.0 {
            return Err(DistributionError::InvalidParameters(
                WishartError::NMustBeGTEDimension.into(),
            ));
        }

        Ok(Self { lv, n })
    }

    pub fn lv(&self) -> &PPTRF {
        &self.lv
    }

    pub fn n(&self) -> f64 {
        self.n
    }
}

impl RandomVariable for WishartParams {
    type RestoreInfo = ();

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        todo!()
    }

    fn restore(v: &[f64], info: Self::RestoreInfo) -> Result<Self, DistributionError> {
        todo!()
    }
}
