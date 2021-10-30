use crate::{Distribution, DistributionError, RandomVariable};
use rand::prelude::*;
use std::{collections::HashSet, hash::Hash, marker::PhantomData};

#[derive(Clone, Debug)]
pub struct DiscreteUniform<T>
where
    T: RandomVariable + Eq + Hash,
{
    phantom: PhantomData<T>,
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
    type T = T;
    type U = HashSet<T>;

    fn p(&self, _: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        Ok(1.0 / theta.len() as f64)
    }

    fn sample(
        &self,
        theta: &Self::U,
        rng: &mut rand::prelude::StdRng,
    ) -> Result<Self::T, DistributionError> {
        let i = rng.gen_range(0..theta.len());

        for (j, x) in theta.iter().enumerate() {
            if i == j {
                return Ok(x.clone());
            }
        }

        panic!("")
    }
}
