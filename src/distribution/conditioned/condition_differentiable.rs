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

pub struct ConditionDifferentiableConditionedDistribution<C, D, T, U1, U2, F, G>
where
    C: ConditionedDistribution<Distribution = D, Condition = F>,
    D: Distribution<Value = T, Condition = U1>,
    T: RandomVariable,
    U1: Event,
    U2: Event,
    F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
    G: Fn(&U2) -> Result<U1, DistributionError>,
{
    conditioned_distribution: C,
    condition_diff: G,
    phantom: PhantomData<U2>,
}

impl<C, D, T, U1, U2, F, G> ConditionDifferentiableConditionedDistribution<C, D, T, U1, U2, F, G>
where
    C: ConditionedDistribution<Distribution = D, Condition = F>,
    D: Distribution<Value = T, Condition = U1>,
    T: RandomVariable,
    U1: Event,
    U2: Event,
    F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
    G: Fn(&U2) -> Result<U1, DistributionError>,
{
    pub fn new(conditioned_distribution: C, condition_diff: G) -> Self {
        Self {
            conditioned_distribution,
            condition_diff,
            phantom: PhantomData,
        }
    }
}
