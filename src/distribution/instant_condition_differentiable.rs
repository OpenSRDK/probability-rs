use crate::{
    ConditionDifferentiableDistribution, DependentJoint, Distribution, DistributionError,
    IndependentJoint, InstantDistribution, RandomVariable, SamplableDistribution,
};
use rand::prelude::*;
use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::{BitAnd, Mul},
};

#[derive(Clone)]
pub struct ConditionDifferentiableInstantDistribution<T, U, FF, FS, G>
where
    T: RandomVariable,
    U: RandomVariable,
    FF: Fn(&T, &U) -> Result<f64, DistributionError> + Clone + Send + Sync,
    FS: Fn(&U, &mut dyn RngCore) -> Result<T, DistributionError> + Clone + Send + Sync,
    G: Fn(&T, &U) -> Result<Vec<f64>, DistributionError> + Clone + Send + Sync,
{
    instant_distribution: InstantDistribution<T, U, FF, FS>,
    condition_diff: G,
    phantom: PhantomData<U>,
}

impl<T, U, FF, FS, G> ConditionDifferentiableInstantDistribution<T, U, FF, FS, G>
where
    T: RandomVariable,
    U: RandomVariable,
    FF: Fn(&T, &U) -> Result<f64, DistributionError> + Clone + Send + Sync,
    FS: Fn(&U, &mut dyn RngCore) -> Result<T, DistributionError> + Clone + Send + Sync,
    G: Fn(&T, &U) -> Result<Vec<f64>, DistributionError> + Clone + Send + Sync,
{
    pub fn new(instant_distribution: InstantDistribution<T, U, FF, FS>, condition_diff: G) -> Self {
        Self {
            instant_distribution,
            condition_diff,
            phantom: PhantomData,
        }
    }
}

impl<T, U, FF, FS, G> Debug for ConditionDifferentiableInstantDistribution<T, U, FF, FS, G>
where
    T: RandomVariable,
    U: RandomVariable,
    FF: Fn(&T, &U) -> Result<f64, DistributionError> + Clone + Send + Sync,
    FS: Fn(&U, &mut dyn RngCore) -> Result<T, DistributionError> + Clone + Send + Sync,
    G: Fn(&T, &U) -> Result<Vec<f64>, DistributionError> + Clone + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InstantDistribution")
    }
}

impl<T, U, FF, FS, G> Distribution for ConditionDifferentiableInstantDistribution<T, U, FF, FS, G>
where
    T: RandomVariable,
    U: RandomVariable,
    FF: Fn(&T, &U) -> Result<f64, DistributionError> + Clone + Send + Sync,
    FS: Fn(&U, &mut dyn RngCore) -> Result<T, DistributionError> + Clone + Send + Sync,
    G: Fn(&T, &U) -> Result<Vec<f64>, DistributionError> + Clone + Send + Sync,
{
    type Value = T;
    type Condition = U;

    fn p_kernel(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<f64, crate::DistributionError> {
        self.instant_distribution.p_kernel(x, theta)
    }
}

impl<T, U, Rhs, TRhs, FF, FS, G> Mul<Rhs>
    for ConditionDifferentiableInstantDistribution<T, U, FF, FS, G>
where
    T: RandomVariable,
    U: RandomVariable,
    Rhs: Distribution<Value = TRhs, Condition = U>,
    TRhs: RandomVariable,
    FF: Fn(&T, &U) -> Result<f64, DistributionError> + Clone + Send + Sync,
    FS: Fn(&U, &mut dyn RngCore) -> Result<T, DistributionError> + Clone + Send + Sync,
    G: Fn(&T, &U) -> Result<Vec<f64>, DistributionError> + Clone + Send + Sync,
{
    type Output = IndependentJoint<Self, Rhs, T, TRhs, U>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<T, U, Rhs, URhs, FF, FS, G> BitAnd<Rhs>
    for ConditionDifferentiableInstantDistribution<T, U, FF, FS, G>
where
    T: RandomVariable,
    U: RandomVariable,
    Rhs: Distribution<Value = U, Condition = URhs>,
    URhs: RandomVariable,
    FF: Fn(&T, &U) -> Result<f64, DistributionError> + Clone + Send + Sync,
    FS: Fn(&U, &mut dyn RngCore) -> Result<T, DistributionError> + Clone + Send + Sync,
    G: Fn(&T, &U) -> Result<Vec<f64>, DistributionError> + Clone + Send + Sync,
{
    type Output = DependentJoint<Self, Rhs, T, U, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

impl<T, U, FF, FS, G> ConditionDifferentiableDistribution
    for ConditionDifferentiableInstantDistribution<T, U, FF, FS, G>
where
    T: RandomVariable,
    U: RandomVariable,
    FF: Fn(&T, &U) -> Result<f64, DistributionError> + Clone + Send + Sync,
    FS: Fn(&U, &mut dyn RngCore) -> Result<T, DistributionError> + Clone + Send + Sync,
    G: Fn(&T, &U) -> Result<Vec<f64>, DistributionError> + Clone + Send + Sync,
{
    fn ln_diff_condition(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let g = (self.condition_diff)(x, theta)?;
        Ok(g)
    }
}

impl<T, U, FF, FS, G> SamplableDistribution
    for ConditionDifferentiableInstantDistribution<T, U, FF, FS, G>
where
    T: RandomVariable,
    U: RandomVariable,
    FF: Fn(&T, &U) -> Result<f64, DistributionError> + Clone + Send + Sync,
    FS: Fn(&U, &mut dyn RngCore) -> Result<T, DistributionError> + Clone + Send + Sync,
    G: Fn(&T, &U) -> Result<Vec<f64>, DistributionError> + Clone + Send + Sync,
{
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, crate::DistributionError> {
        self.instant_distribution.sample(theta, rng)
    }
}
