use crate::{
    DiscreteDistribution, Distribution, DistributionError, RandomVariable, SampleableDistribution,
};
use rand::prelude::*;
use std::{collections::HashSet, hash::Hash, marker::PhantomData};

#[derive(Clone, Debug)]
pub struct DiscreteUniform<T>
where
    T: RandomVariable + Eq + Hash,
{
    phantom: PhantomData<T>,
}

#[derive(thiserror::Error, Debug)]
pub enum DiscreteUniformError {
    #[error("Range is empty.")]
    RangeIsEmpty,
    #[error("Unknown error")]
    Unknown,
}

impl<T> DiscreteUniform<T>
where
    T: RandomVariable + Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<T> Distribution for DiscreteUniform<T>
where
    T: RandomVariable + Eq + Hash,
{
    type Value = T;
    type Condition = HashSet<T>;

    fn fk(&self, _x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        Ok(1.0 / theta.len() as f64)
    }
}

impl<T> DiscreteDistribution for DiscreteUniform<T> where T: RandomVariable + Eq + Hash {}

impl<T> SampleableDistribution for DiscreteUniform<T>
where
    T: RandomVariable + Eq + Hash,
{
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let len = theta.len();
        if len == 0 {
            return Err(DistributionError::InvalidParameters(
                DiscreteUniformError::RangeIsEmpty.into(),
            ));
        }
        let i = rng.gen_range(0..len);

        for (j, x) in theta.iter().enumerate() {
            if i == j {
                return Ok(x.clone());
            }
        }

        Err(DistributionError::InvalidParameters(
            DiscreteUniformError::Unknown.into(),
        ))
    }
}
