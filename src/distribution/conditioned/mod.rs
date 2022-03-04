pub mod condition_differentiable;

pub use condition_differentiable::*;

use crate::{
    DependentJoint, Distribution, DistributionError, Event, IndependentJoint, RandomVariable,
    ValueDifferentiableDistribution,
};
use rand::prelude::*;
use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::{BitAnd, Mul},
};

#[derive(Clone)]
pub struct ConditionedDistribution<D, T, U1, U2, F>
where
    D: Distribution<Value = T, Condition = U1>,
    T: RandomVariable,
    U1: Event,
    U2: Event,
    F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
{
    distribution: D,
    condition: F,
    phantom: PhantomData<U2>,
}

impl<D, T, U1, U2, F> ConditionedDistribution<D, T, U1, U2, F>
where
    D: Distribution<Value = T, Condition = U1>,
    T: RandomVariable,
    U1: Event,
    U2: Event,
    F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
{
    pub fn new(distribution: D, condition: F) -> Self {
        Self {
            distribution,
            condition,
            phantom: PhantomData,
        }
    }
}

impl<D, T, U1, U2, F> Debug for ConditionedDistribution<D, T, U1, U2, F>
where
    D: Distribution<Value = T, Condition = U1>,
    T: RandomVariable,
    U1: Event,
    U2: Event,
    F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ConditionedDistribution {{ distribution: {:#?} }}",
            self.distribution
        )
    }
}

impl<D, T, U1, U2, F> Distribution for ConditionedDistribution<D, T, U1, U2, F>
where
    D: Distribution<Value = T, Condition = U1>,
    T: RandomVariable,
    U1: Event,
    U2: Event,
    F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
{
    type Value = T;
    type Condition = U2;

    fn fk(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<f64, crate::DistributionError> {
        self.distribution.fk(x, &(self.condition)(theta)?)
    }

    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, crate::DistributionError> {
        self.distribution.sample(&(self.condition)(theta)?, rng)
    }
}

pub trait ConditionableDistribution: Distribution + Sized {
    /// .
    ///
    /// # Examples
    ///
    /// ```
    /// // Example template not implemented for trait functions
    /// ```
    fn condition<U2, F>(
        self,
        condition: F,
    ) -> ConditionedDistribution<Self, Self::Value, Self::Condition, U2, F>
    where
        U2: Event,
        F: Fn(&U2) -> Result<Self::Condition, DistributionError> + Clone + Send + Sync;
}

impl<D, T, U1> ConditionableDistribution for D
where
    D: Distribution<Value = T, Condition = U1>,
    T: RandomVariable,
    U1: Event,
{
    fn condition<U2, F>(
        self,
        condition: F,
    ) -> ConditionedDistribution<Self, Self::Value, Self::Condition, U2, F>
    where
        U2: Event,
        F: Fn(&U2) -> Result<Self::Condition, DistributionError> + Clone + Send + Sync,
    {
        ConditionedDistribution::<Self, Self::Value, Self::Condition, U2, F>::new(self, condition)
    }
}

impl<D, T, U1, U2, Rhs, TRhs, F> Mul<Rhs> for ConditionedDistribution<D, T, U1, U2, F>
where
    D: Distribution<Value = T, Condition = U1>,
    T: RandomVariable,
    U1: Event,
    U2: Event,
    Rhs: Distribution<Value = TRhs, Condition = U2>,
    TRhs: RandomVariable,
    F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
{
    type Output = IndependentJoint<Self, Rhs, T, TRhs, U2>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<D, T, U1, U2, Rhs, URhs, F> BitAnd<Rhs> for ConditionedDistribution<D, T, U1, U2, F>
where
    D: Distribution<Value = T, Condition = U1>,
    T: RandomVariable,
    U1: Event,
    U2: Event,
    Rhs: Distribution<Value = U2, Condition = URhs>,
    URhs: RandomVariable,
    F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
{
    type Output = DependentJoint<Self, Rhs, T, U2, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

impl<D, T, U1, U2, F> ValueDifferentiableDistribution for ConditionedDistribution<D, T, U1, U2, F>
where
    D: Distribution<Value = T, Condition = U1> + ValueDifferentiableDistribution,
    T: RandomVariable,
    U1: Event,
    U2: Event,
    F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
{
    fn ln_diff_value(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let f = self
            .distribution
            .ln_diff_value(x, &(self.condition)(theta)?)
            .unwrap();
        Ok(f)
    }
}
