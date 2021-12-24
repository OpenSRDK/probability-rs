use std::{fmt::Debug, marker::PhantomData};

use crate::{Distribution, RandomVariable};

#[derive(Clone)]
pub struct Degenerate<T, U, F>
where
    T: RandomVariable + PartialEq,
    U: RandomVariable,
    F: Fn(&U) -> T + Clone + Send + Sync,
{
    f: F,
    phantom: PhantomData<(T, U)>,
}

impl<T, U, F> Degenerate<T, U, F>
where
    T: RandomVariable + PartialEq,
    U: RandomVariable,
    F: Fn(&U) -> T + Clone + Send + Sync,
{
    pub fn new(f: F) -> Self {
        Self {
            f,
            phantom: PhantomData,
        }
    }
}

impl<T, U, F> Debug for Degenerate<T, U, F>
where
    T: RandomVariable + PartialEq,
    U: RandomVariable,
    F: Fn(&U) -> T + Clone + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DegenerateDistribution {{ }}",)
    }
}

impl<T, U, F> Distribution for Degenerate<T, U, F>
where
    T: RandomVariable + PartialEq,
    U: RandomVariable,
    F: Fn(&U) -> T + Clone + Send + Sync,
{
    type Value = T;
    type Condition = U;

    fn fk(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<f64, crate::DistributionError> {
        let value = (self.f)(theta);
        if value.eq(x) {
            Ok(1.0)
        } else {
            Ok(0.0)
        }
    }

    fn sample(
        &self,
        theta: &Self::Condition,
        _rng: &mut dyn rand::RngCore,
    ) -> Result<Self::Value, crate::DistributionError> {
        Ok((self.f)(theta))
    }
}
