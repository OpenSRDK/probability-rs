use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::{BitAnd, Mul},
};

#[derive(Clone, Debug)]
pub struct ConditionalizeLatentVariableDistribution<D, F, T, U, V, W>
where
    D: Distribution<Value = T, Condition = U>,
    F: Fn((V, W)) -> (T, U),
    T: RandomVariable,
    U: RandomVariable,
    V: RandomVariable,
    W: RandomVariable,
{
    distribution: D,
    converter: F,
    phantom: PhantomData<(V, W)>,
}

impl<D, F, T, U, V, W> ConditionalizeLatentVariableDistribution<D, F, T, U, V, W>
where
    D: Distribution<Value = T, Condition = U>,
    F: Fn((V, W)) -> (T, U),
    T: RandomVariable,
    U: RandomVariable,
    V: RandomVariable,
    W: RandomVariable,
{
    pub fn new(distribution: D, converter: F) -> Self {
        Self {
            distribution,
            converter,
            phantom: PhantomData,
        }
    }
}

impl<D, F, T, U, V, W> Distribution for ConditionalizeLatentVariableDistribution<D, F, T, U, V, W>
where
    D: Distribution<Value = T, Condition = U>,
    F: Fn((V, W)) -> (T, U),
    T: RandomVariable,
    U: RandomVariable,
    V: RandomVariable,
    W: RandomVariable,
{
    type Value = V;
    type Condition = W;

    fn fk(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<f64, crate::DistributionError> {
        self.distribution.fk((x, theta).converter())
    }

    fn sample(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, crate::DistributionError> {
        todo!()
    }
}

impl<D, F, T, U, V, W, Rhs, TRhs> Mul<Rhs>
    for ConditionalizeLatentVariableDistribution<D, F, T, U, V, W>
where
    D: Distribution<Value = T, Condition = U>,
    F: Fn((V, W)) -> (T, U),
    T: RandomVariable,
    U: RandomVariable,
    V: RandomVariable,
    W: RandomVariable,
    Rhs: Distribution<Value = TRhs, Condition = U>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, T, TRhs, U>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<D, F, T, U, V, W, Rhs, TRhs> Mul<Rhs>
    for ConditionalizeLatentVariableDistribution<D, F, T, U, V, W>
where
    D: Distribution<Value = T, Condition = U>,
    F: Fn((V, W)) -> (T, U),
    T: RandomVariable,
    U: RandomVariable,
    V: RandomVariable,
    W: RandomVariable,
    Rhs: Distribution<Value = T, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, T, U, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}
