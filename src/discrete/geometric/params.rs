use crate::{DistributionError, GeometricError, RandomVariable};

#[derive(Clone, Debug, PartialEq)]
pub struct GeometricParams {
    p: f64,
}

impl GeometricParams {
    pub fn new(p: f64) -> Result<Self, DistributionError> {
        if p < 0.0 || 1.0 < p {
            return Err(DistributionError::InvalidParameters(
                GeometricError::PMustBeProbability.into(),
            ));
        }

        Ok(Self { p })
    }

    pub fn p(&self) -> f64 {
        self.p
    }
}

impl RandomVariable for GeometricParams {
    type RestoreInfo = ();

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        (vec![self.p], ())
    }

    fn len(&self) -> usize {
        1usize
    }

    fn restore(v: &[f64], _: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        if v.len() != 1 {
            return Err(DistributionError::InvalidRestoreVector);
        }
        GeometricParams::new(v[0])
    }
}
