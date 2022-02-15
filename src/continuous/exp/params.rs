use crate::{DistributionError, ExpError, RandomVariable};

#[derive(Clone, Debug, PartialEq)]
pub struct ExpParams {
    lambda: f64,
}

impl ExpParams {
    pub fn new(lambda: f64) -> Result<Self, DistributionError> {
        if lambda <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                ExpError::LambdaMustBePositive.into(),
            ));
        }

        Ok(Self { lambda })
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl RandomVariable for ExpParams {
    type RestoreInfo = ();

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        (vec![self.lambda], ())
    }

    fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        Self::new(v[0])
    }
}
