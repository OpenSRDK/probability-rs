use crate::{DistributionError, RandomVariable, WishartError};
use opensrdk_linear_algebra::{pp::trf::PPTRF, SymmetricPackedMatrix};

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
    type RestoreInfo = usize;

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        let p = self.lv.0.dim();
        ([self.lv.0.elems(), &[self.n]].concat(), p)
    }

    fn len(&self) -> usize {
        let t = self.lv.0.elems().len();
        t + 1usize
    }

    fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        if v.len() != info + 1 {
            return Err(DistributionError::InvalidRestoreVector);
        }
        let p = *info;
        let n = v[v.len() - 1];
        let lv = PPTRF(SymmetricPackedMatrix::from(p, v[0..p].to_vec()).unwrap());
        Self::new(lv, n)
    }
}
