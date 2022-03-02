use crate::{DistributionError, PoissonError, RandomVariable};

#[derive(Clone, Debug, PartialEq)]
pub struct PoissonParams {
    lambda: f64,
}

impl PoissonParams {
    pub fn new(lambda: f64) -> Result<Self, DistributionError> {
        if lambda <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                PoissonError::LambdaMustBePositive.into(),
            ));
        }

        Ok(Self { lambda })
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl RandomVariable for PoissonParams {
    type RestoreInfo = ();

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        (vec![self.lambda], ())
    }

    fn len(&self) -> usize {
        todo!()
    }

    fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        if v.len() != 1 {
            return Err(DistributionError::InvalidRestoreVector);
        }
        PoissonParams::new(v[0])
    }
}
