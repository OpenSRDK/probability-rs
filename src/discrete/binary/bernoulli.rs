use crate::{Distribution, DistributionError};

#[derive(Clone, Debug)]
pub struct Bernoulli;

#[derive(thiserror::Error, Debug)]
pub enum BernoulliError {
    #[error("'p' must be probability.")]
    PMustBeProbability,
}

impl Distribution for Bernoulli {
    type T = bool;
    type U = f64;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        todo!()
    }

    fn sample(
        &self,
        theta: &Self::U,
        rng: &mut dyn rand::RngCore,
    ) -> Result<Self::T, DistributionError> {
        todo!()
    }
}

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
