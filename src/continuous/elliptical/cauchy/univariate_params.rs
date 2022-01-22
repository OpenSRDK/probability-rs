use crate::DistributionError;
use crate::{CauchyError, RandomVariable};

#[derive(Clone, Debug)]
pub struct CauchyParams {
    mu: f64,
    sigma: f64,
}

impl CauchyParams {
    pub fn new(mu: f64, sigma: f64) -> Result<Self, DistributionError> {
        if sigma <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                CauchyError::SigmaMustBePositive.into(),
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

impl RandomVariable for CauchyParams {
    type RestoreInfo = ();

    fn transform_vec(self) -> (Vec<f64>, Self::RestoreInfo) {
        (vec![self.mu, self.sigma], ())
    }

    fn restore(v: Vec<f64>, _: Self::RestoreInfo) -> Self {
        Self::new(v[0], v[1]).unwrap()
    }
}
