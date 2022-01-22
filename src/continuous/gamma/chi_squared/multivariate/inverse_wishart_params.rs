use opensrdk_linear_algebra::pp::trf::PPTRF;

use crate::{DistributionError, InverseWishartError, RandomVariable};

#[derive(Clone, Debug, PartialEq)]
pub struct InverseWishartParams {
    lpsi: PPTRF,
    nu: f64,
}

impl InverseWishartParams {
    pub fn new(lpsi: PPTRF, nu: f64) -> Result<Self, DistributionError> {
        let p = lpsi.0.dim();

        if nu <= p as f64 - 1.0 {
            return Err(DistributionError::InvalidParameters(
                InverseWishartError::NuMustBeGTEDimension.into(),
            ));
        }

        Ok(Self { lpsi, nu })
    }

    pub fn lpsi(&self) -> &PPTRF {
        &self.lpsi
    }

    pub fn nu(&self) -> f64 {
        self.nu
    }
}

impl RandomVariable for InverseWishartParams {
    type RestoreInfo = usize;

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        todo!()
    }

    fn restore(v: &[f64], info: Self::RestoreInfo) -> Result<Self, DistributionError> {
        todo!()
    }
}
