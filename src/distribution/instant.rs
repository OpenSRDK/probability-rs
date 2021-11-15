use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use std::{
    fmt::Debug,
    ops::{BitAnd, Mul},
};

#[derive(Clone)]
pub struct InstantDistribution<'a, T, U>
where
    T: RandomVariable,
    U: RandomVariable,
{
    p: &'a (dyn Fn(&T, &U) -> Result<f64, DistributionError> + Send + Sync),
    sample: &'a (dyn Fn(&U, &mut dyn RngCore) -> Result<T, DistributionError> + Send + Sync),
}

impl<'a, T, U> InstantDistribution<'a, T, U>
where
    T: RandomVariable,
    U: RandomVariable,
{
    pub fn new(
        p: &'a (dyn Fn(&T, &U) -> Result<f64, DistributionError> + Send + Sync),
        sample: &'a (dyn Fn(&U, &mut dyn RngCore) -> Result<T, DistributionError> + Send + Sync),
    ) -> Self {
        Self { p, sample }
    }
}

impl<'a, T, U> Debug for InstantDistribution<'a, T, U>
where
    T: RandomVariable,
    U: RandomVariable,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Instant")
    }
}

impl<'a, T, U> Distribution for InstantDistribution<'a, T, U>
where
    T: RandomVariable,
    U: RandomVariable,
{
    type Value = T;
    type Condition = U;

    fn fk(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        (self.p)(x, theta)
    }

    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        (self.sample)(theta, rng)
    }
}

impl<'a, T, U, Rhs, TRhs> Mul<Rhs> for InstantDistribution<'a, T, U>
where
    T: RandomVariable,
    U: RandomVariable,
    Rhs: Distribution<Value = TRhs, Condition = U>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, T, TRhs, U>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<'a, T, U, Rhs, URhs> BitAnd<Rhs> for InstantDistribution<'a, T, U>
where
    T: RandomVariable,
    U: RandomVariable,
    Rhs: Distribution<Value = U, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, T, U, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}
