use crate::{Distribution, DistributionError};

#[derive(Clone, Debug)]
pub struct Categorical;

#[derive(thiserror::Error, Debug)]
pub enum CategoricalError {
    #[error("'p' must be probability.")]
    PMustBeProbability,
}

impl Distribution for Categorical {
    type T = usize;
    type U = Vec<f64>;

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
pub struct CategoricalParams {
    p: Vec<f64>,
}

impl CategoricalParams {
    pub fn new(p: Vec<f64>) -> Result<Self, DistributionError> {
        for &pi in p.iter() {
            if pi < 0.0 || 1.0 < pi {
                return Err(DistributionError::InvalidParameters(
                    CategoricalError::PMustBeProbability.into(),
                ));
            }
        }

        Ok(Self { p })
    }

    pub fn p(&self) -> &Vec<f64> {
        &self.p
    }
}
