pub mod conditionalize_latent;

pub use conditionalize_latent::*;

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

impl<D, F, T, U, V, W> ValueDifferentiableDistribution
    for ConditionalizeLatentVariableDistribution<D, F, T, U, V, W>
where
    D: Distribution<Value = T, Condition = U>,
    F: Fn((V, W)) -> (T, U),
    T: RandomVariable,
    U: RandomVariable,
    V: RandomVariable,
    W: RandomVariable,
{
    fn ln_diff_value(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let converted = (x, theta).converter()?;
        let v_len = x.len();
        let t_len = converted.0.len();
        let diff = t_len - v_len;
        if 0 < diff {
            let value_orig = self.distribution.ln_diff_value(converted)?;
            let condition_orig = self.distribution.ln_diff_condition(converted)?;
            let result_value = concat![value_orig, condition_orig[..diff]];
            Ok(result_value)
        } else {
            let value_orig = self.distribution.ln_diff_value(converted)?;
            Ok(result_value[..t_len])
        }
    }
}

impl<D, F, T, U, V, W> ConditionDifferentiableDistribution
    for ConditionalizeLatentVariableDistribution<D, F, T, U, V, W>
where
    D: Distribution<Value = T, Condition = U>,
    F: Fn((V, W)) -> (T, U),
    T: RandomVariable,
    U: RandomVariable,
    V: RandomVariable,
    W: RandomVariable,
{
    fn ln_diff_condition(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let converted = (x, theta).converter()?;
        let w_len = x.len();
        let u_len = converted.1.len();
        let diff = u_len - w_len;
        if 0 < diff {
            let value_orig = self.distribution.ln_diff_value(converted)?;
            let condition_orig = self.distribution.ln_diff_condition(converted)?;
            let result_condition = concat![value_orig[w_len..], condition_orig];
            Ok(result_condition)
        } else {
            let value_orig = self.distribution.ln_diff_condition(converted)?;
            Ok(result_condition[..w_len])
        }
    }
}
