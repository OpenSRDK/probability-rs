use crate::{
    DependentJoint, Distribution, DistributionError, IndependentJoint, InstantDistribution,
    RandomVariable, SamplableDistribution, ValueDifferentiableDistribution,
};
use rand::prelude::*;
use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::{BitAnd, Mul},
};

#[derive(Clone)]
pub struct ValueDifferentiableInstantDistribution<T, U, FF, FS, G>
where
    T: RandomVariable,
    U: RandomVariable,
    FF: Fn(&T, &U) -> Result<f64, DistributionError> + Clone + Send + Sync,
    FS: Fn(&U, &mut dyn RngCore) -> Result<T, DistributionError> + Clone + Send + Sync,
    G: Fn(&T, &U) -> Result<Vec<f64>, DistributionError> + Clone + Send + Sync,
{
    instant_distribution: InstantDistribution<T, U, FF, FS>,
    value_diff: G,
    phantom: PhantomData<U>,
}

impl<T, U, FF, FS, G> ValueDifferentiableInstantDistribution<T, U, FF, FS, G>
where
    T: RandomVariable,
    U: RandomVariable,
    FF: Fn(&T, &U) -> Result<f64, DistributionError> + Clone + Send + Sync,
    FS: Fn(&U, &mut dyn RngCore) -> Result<T, DistributionError> + Clone + Send + Sync,
    G: Fn(&T, &U) -> Result<Vec<f64>, DistributionError> + Clone + Send + Sync,
{
    pub fn new(instant_distribution: InstantDistribution<T, U, FF, FS>, value_diff: G) -> Self {
        Self {
            instant_distribution,
            value_diff,
            phantom: PhantomData,
        }
    }
}

impl<T, U, FF, FS, G> Debug for ValueDifferentiableInstantDistribution<T, U, FF, FS, G>
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

impl<T, U, FF, FS, G> Distribution for ValueDifferentiableInstantDistribution<T, U, FF, FS, G>
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
    for ValueDifferentiableInstantDistribution<T, U, FF, FS, G>
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
    for ValueDifferentiableInstantDistribution<T, U, FF, FS, G>
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

impl<T, U, FF, FS, G> ValueDifferentiableDistribution
    for ValueDifferentiableInstantDistribution<T, U, FF, FS, G>
where
    T: RandomVariable,
    U: RandomVariable,
    FF: Fn(&T, &U) -> Result<f64, DistributionError> + Clone + Send + Sync,
    FS: Fn(&U, &mut dyn RngCore) -> Result<T, DistributionError> + Clone + Send + Sync,
    G: Fn(&T, &U) -> Result<Vec<f64>, DistributionError> + Clone + Send + Sync,
{
    fn ln_diff_value(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let g = (self.value_diff)(x, theta)?;
        Ok(g)
    }
}

impl<T, U, FF, FS, G> SamplableDistribution
    for ValueDifferentiableInstantDistribution<T, U, FF, FS, G>
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
