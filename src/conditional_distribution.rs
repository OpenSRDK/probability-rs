use crate::Distribution;
use rand::prelude::StdRng;
use std::error::Error;

#[derive(thiserror::Error, Debug)]
pub enum ConditionalError {
    #[error("not conditioned")]
    NotConditioned,
}

pub struct ConditionalDistribution<T, U> {
    p: Box<dyn Fn(T) -> Result<f64, Box<dyn Error>>>,
    sample: Box<dyn Fn(&mut StdRng) -> Result<T, Box<dyn Error>>>,
    condition: Option<U>,
}

impl<T, U> ConditionalDistribution<T, U> {
    pub fn new(
        p: Box<dyn Fn(T) -> Result<f64, Box<dyn Error>>>,
        sample: Box<dyn Fn(&mut StdRng) -> Result<T, Box<dyn Error>>>,
    ) -> Self {
        Self {
            p,
            sample,
            condition: None,
        }
    }

    pub fn with_condition(&mut self, condition: U) -> &mut Self {
        self.condition = Some(condition);

        self
    }
}

impl<T, U> Distribution<T> for ConditionalDistribution<T, U> {
    fn p(&self, x: T) -> Result<f64, Box<dyn Error>> {
        if self.condition.is_none() {
            return Err(ConditionalError::NotConditioned.into());
        }

        (self.p)(x)
    }

    fn sample(&self, rng: &mut StdRng) -> Result<T, Box<dyn Error>> {
        if self.condition.is_none() {
            return Err(ConditionalError::NotConditioned.into());
        }

        (self.sample)(rng)
    }
}
