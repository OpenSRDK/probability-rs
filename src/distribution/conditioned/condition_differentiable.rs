use crate::{
    ConditionDifferentiableDistribution, ConditionedDistribution, DependentJoint, Distribution,
    DistributionError, Event, IndependentJoint, RandomVariable, ValueDifferentiableDistribution,
};
use rand::prelude::*;
use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::{BitAnd, Mul},
};

#[derive(Clone)]
pub struct ConditionDifferentiableConditionedDistribution<D, T, U1, U2, F, G>
where
    D: Distribution<Value = T, Condition = U1>,
    T: RandomVariable,
    U1: Event,
    U2: Event,
    F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
    G: Fn(&U2) -> Result<Vec<f64>, DistributionError> + Clone + Send + Sync,
{
    conditioned_distribution: ConditionedDistribution<D, T, U1, U2, F>,
    condition_diff: G,
    phantom: PhantomData<U2>,
}

impl<D, T, U1, U2, F, G> ConditionDifferentiableConditionedDistribution<D, T, U1, U2, F, G>
where
    D: Distribution<Value = T, Condition = U1>,
    T: RandomVariable,
    U1: Event,
    U2: Event,
    F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
    G: Fn(&U2) -> Result<Vec<f64>, DistributionError> + Clone + Send + Sync,
{
    pub fn new(
        conditioned_distribution: ConditionedDistribution<D, T, U1, U2, F>,
        condition_diff: G,
    ) -> Self {
        Self {
            conditioned_distribution,
            condition_diff,
            phantom: PhantomData,
        }
    }
}

impl<D, T, U1, U2, F, G> Debug
    for ConditionDifferentiableConditionedDistribution<D, T, U1, U2, F, G>
where
    D: Distribution<Value = T, Condition = U1>,
    T: RandomVariable,
    U1: Event,
    U2: Event,
    F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
    G: Fn(&U2) -> Result<Vec<f64>, DistributionError> + Clone + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ConditionedDistribution {{ distribution: {:#?} }}",
            self.conditioned_distribution.distribution
        )
    }
}

impl<D, T, U1, U2, F, G> Distribution
    for ConditionDifferentiableConditionedDistribution<D, T, U1, U2, F, G>
where
    D: Distribution<Value = T, Condition = U1>,
    T: RandomVariable,
    U1: Event,
    U2: Event,
    F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
    G: Fn(&U2) -> Result<Vec<f64>, DistributionError> + Clone + Send + Sync,
{
    type Value = T;
    type Condition = U2;

    fn fk(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<f64, crate::DistributionError> {
        self.conditioned_distribution
            .distribution
            .fk(x, &(self.conditioned_distribution.condition)(theta)?)
    }

    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, crate::DistributionError> {
        self.conditioned_distribution
            .distribution
            .sample(&(self.conditioned_distribution.condition)(theta)?, rng)
    }
}

impl<D, T, U1, U2, Rhs, TRhs, F, G> Mul<Rhs>
    for ConditionDifferentiableConditionedDistribution<D, T, U1, U2, F, G>
where
    D: Distribution<Value = T, Condition = U1>,
    T: RandomVariable,
    U1: Event,
    U2: Event,
    Rhs: Distribution<Value = TRhs, Condition = U2>,
    TRhs: RandomVariable,
    F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
    G: Fn(&U2) -> Result<Vec<f64>, DistributionError> + Clone + Send + Sync,
{
    type Output = IndependentJoint<Self, Rhs, T, TRhs, U2>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<D, T, U1, U2, Rhs, URhs, F, G> BitAnd<Rhs>
    for ConditionDifferentiableConditionedDistribution<D, T, U1, U2, F, G>
where
    D: Distribution<Value = T, Condition = U1>,
    T: RandomVariable,
    U1: Event,
    U2: Event,
    Rhs: Distribution<Value = U2, Condition = URhs>,
    URhs: RandomVariable,
    F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
    G: Fn(&U2) -> Result<Vec<f64>, DistributionError> + Clone + Send + Sync,
{
    type Output = DependentJoint<Self, Rhs, T, U2, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

impl<D, T, U1, U2, F, G> ValueDifferentiableDistribution
    for ConditionDifferentiableConditionedDistribution<D, T, U1, U2, F, G>
where
    D: Distribution<Value = T, Condition = U1> + ValueDifferentiableDistribution,
    T: RandomVariable,
    U1: Event,
    U2: Event,
    F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
    G: Fn(&U2) -> Result<Vec<f64>, DistributionError> + Clone + Send + Sync,
{
    fn ln_diff_value(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let f = self
            .conditioned_distribution
            .distribution
            .ln_diff_value(x, &(self.conditioned_distribution.condition)(theta)?)
            .unwrap();
        Ok(f)
    }
}

impl<D, T, U1, U2, F, G> ConditionDifferentiableDistribution
    for ConditionDifferentiableConditionedDistribution<D, T, U1, U2, F, G>
where
    D: Distribution<Value = T, Condition = U1> + ConditionDifferentiableDistribution,
    T: RandomVariable,
    U1: Event,
    U2: Event,
    F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
    G: Fn(&U2) -> Result<Vec<f64>, DistributionError> + Clone + Send + Sync,
{
    fn ln_diff_condition(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let f = self
            .conditioned_distribution
            .distribution
            .ln_diff_condition(x, &(self.conditioned_distribution.condition)(theta)?)
            .unwrap();
        let g = &(self.condition_diff)(theta)?;

        let diff = f
            .iter()
            .enumerate()
            .map(|(i, fi)| fi * g[i])
            .collect::<Vec<f64>>();
        Ok(diff)
    }
}
