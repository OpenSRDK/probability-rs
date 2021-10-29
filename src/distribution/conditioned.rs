use crate::{DependentJoint, Distribution, DistributionError, IndependentJoint, RandomVariable};
use std::{
    fmt::Debug,
    ops::{BitAnd, Mul},
};

#[derive(Clone)]
pub struct ConditionedDistribution<'a, D, T, U1, U2>
where
    D: Distribution<T = T, U = U1>,
    T: RandomVariable,
    U1: RandomVariable,
    U2: RandomVariable,
{
    distribution: D,
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
        distribution: D,
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

pub trait ConditionableDistribution: Distribution + Sized {
    fn condition<'a, U2>(
        self,
        condition: &'a (dyn Fn(&U2) -> Result<Self::U, DistributionError> + Send + Sync),
    ) -> ConditionedDistribution<'a, Self, Self::T, Self::U, U2>
    where
        U2: RandomVariable;
}

impl<D, T, U1> ConditionableDistribution for D
where
    D: Distribution<T = T, U = U1>,
    T: RandomVariable,
    U1: RandomVariable,
{
    fn condition<'a, U2>(
        self,
        condition: &'a (dyn Fn(&U2) -> Result<Self::U, DistributionError> + Send + Sync),
    ) -> ConditionedDistribution<'a, Self, Self::T, Self::U, U2>
    where
        U2: RandomVariable,
    {
        ConditionedDistribution::<Self, Self::T, Self::U, U2>::new(self, condition)
    }
}

impl<'a, D, T, U1, U2, Rhs, TRhs> Mul<Rhs> for ConditionedDistribution<'a, D, T, U1, U2>
where
    D: Distribution<T = T, U = U1>,
    T: RandomVariable,
    U1: RandomVariable,
    U2: RandomVariable,
    Rhs: Distribution<T = TRhs, U = U2>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, T, TRhs, U2>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<'a, D, T, U1, U2, Rhs, URhs> BitAnd<Rhs> for ConditionedDistribution<'a, D, T, U1, U2>
where
    D: Distribution<T = T, U = U1>,
    T: RandomVariable,
    U1: RandomVariable,
    U2: RandomVariable,
    Rhs: Distribution<T = U2, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, T, U2, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}
