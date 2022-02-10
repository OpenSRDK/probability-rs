use crate::{DistributionError, RandomVariable};

#[derive(Clone, Debug)]
pub struct CategoricalParams {
    p: Vec<f64>,
}

impl CategoricalParams {
    pub fn new(p: Vec<f64>) -> Result<Self, DistributionError> {
        Ok(Self { p })
    }

    pub fn p(&self) -> &Vec<f64> {
        &self.p
    }
}

impl RandomVariable for CategoricalParams {
    type RestoreInfo = usize;

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        (self.p.clone(), self.p.len())
    }

    fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        if v.len() != *info {
            return Err(DistributionError::InvalidRestoreVector);
        }
        CategoricalParams::new(v.to_vec())
    }
}
