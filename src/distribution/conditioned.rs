use crate::{Distribution, DistributionError, RandomVariable};
use std::fmt::Debug;

#[derive(Clone)]
pub struct ConditionedDistribution<'a, D, T, U1, U2>
where
    D: Distribution<T = T, U = U1>,
    T: RandomVariable,
    U1: RandomVariable,
    U2: RandomVariable,
{
    distribution: &'a D,
    condition: &'a (dyn Fn(&U2) -> Result<U1, DistributionError> + Send + Sync),
}

impl<'a, D, T, U1, U2> ConditionedDistribution<'a, D, T, U1, U2>
where
    D: Distribution<T = T, U = U1>,
    T: RandomVariable,
    U1: RandomVariable,
    U2: RandomVariable,
{
    pub fn new(
        distribution: &'a D,
        condition: &'a (dyn Fn(&U2) -> Result<U1, DistributionError> + Send + Sync),
    ) -> Self {
        Self {
            distribution,
            condition,
        }
    }
}

impl<'a, D, T, U1, U2> Debug for ConditionedDistribution<'a, D, T, U1, U2>
where
    D: Distribution<T = T, U = U1>,
    T: RandomVariable,
    U1: RandomVariable,
    U2: RandomVariable,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ConditionedDistribution {{ distribution: {:#?} }}",
            self.distribution
        )
    }
}

impl<'a, D, T, U1, U2> Distribution for ConditionedDistribution<'a, D, T, U1, U2>
where
    D: Distribution<T = T, U = U1>,
    T: RandomVariable,
    U1: RandomVariable,
    U2: RandomVariable,
{
    type T = T;
    type U = U2;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, crate::DistributionError> {
        self.distribution.p(x, &(self.condition)(theta)?)
    }

    fn sample(
        &self,
        theta: &Self::U,
        rng: &mut rand::prelude::StdRng,
    ) -> Result<Self::T, crate::DistributionError> {
        self.distribution.sample(&(self.condition)(theta)?, rng)
    }
}
