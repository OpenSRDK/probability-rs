use crate::{BernoulliError, DistributionError, RandomVariable};

#[derive(Clone, Debug)]
pub struct BernoulliParams {
    p: f64,
}

impl BernoulliParams {
    pub fn new(p: f64) -> Result<Self, DistributionError> {
        if p < 0.0 || 1.0 < p {
            return Err(DistributionError::InvalidParameters(
                BernoulliError::PMustBeProbability.into(),
            ));
        }

        Ok(Self { p })
    }

    pub fn p(&self) -> f64 {
        self.p
    }
}

impl RandomVariable for BernoulliParams {
    type RestoreInfo = ();

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        (vec![self.p], ())
    }

    fn len(&self) -> usize {
        1usize
    }

    fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        if v.len() != 1 {
            return Err(DistributionError::InvalidRestoreVector);
        }
        BernoulliParams::new(v[0])
    }
}
