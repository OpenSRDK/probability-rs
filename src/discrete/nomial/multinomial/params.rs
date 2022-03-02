use crate::{DistributionError, MultinominalError, RandomVariable};

#[derive(Clone, Debug, PartialEq)]
pub struct MultinomialParams {
    n: u64,
    p: f64,
}

impl MultinomialParams {
    pub fn new(n: u64, p: f64) -> Result<Self, DistributionError> {
        if p < 0.0 || 1.0 < p {
            return Err(DistributionError::InvalidParameters(
                MultinominalError::PMustBeProbability.into(),
            ));
        }

        Ok(Self { n, p })
    }

    pub fn n(&self) -> u64 {
        self.n
    }

    pub fn p(&self) -> f64 {
        self.p
    }
}

impl RandomVariable for MultinomialParams {
    type RestoreInfo = u64;

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        (vec![self.p], self.n)
    }

    fn len(&self) -> usize {
        todo!()
    }

    fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        if v.len() != 1 {
            return Err(DistributionError::InvalidRestoreVector);
        }
        MultinomialParams::new(*info, v[1])
    }
}
