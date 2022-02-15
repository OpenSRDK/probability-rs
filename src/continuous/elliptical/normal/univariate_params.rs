use crate::{DistributionError, NormalError, RandomVariable};

#[derive(Clone, Debug, PartialEq)]
pub struct NormalParams {
    mu: f64,
    sigma: f64,
}

impl NormalParams {
    pub fn new(mu: f64, sigma: f64) -> Result<Self, DistributionError> {
        if sigma <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                NormalError::SigmaMustBePositive.into(),
            ));
        }

        Ok(Self { mu, sigma })
    }

    pub fn mu(&self) -> f64 {
        self.mu
    }

    pub fn sigma(&self) -> f64 {
        self.sigma
    }
}

impl Default for NormalParams {
    fn default() -> Self {
        Self::new(0.0, 1.0).unwrap()
    }
}

impl RandomVariable for NormalParams {
    type RestoreInfo = ();

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        (vec![self.mu, self.sigma], ())
    }

    fn restore(v: &[f64], _: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        Self::new(v[0], v[1])
    }
}
