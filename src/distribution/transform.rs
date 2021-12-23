use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::{BitAnd, Mul},
};

#[derive(Clone, Debug)]
pub struct TransformedDistribution<D, T, U, V>
where
    D: Distribution<Value = T, Condition = U>,
    T: RandomVariable,
    U: RandomVariable,
    V: RandomVariable,
{
    distribution: D,
    phantom: PhantomData<V>,
}

impl<D, T, U, V> TransformedDistribution<D, T, U, V>
where
    D: Distribution<Value = T, Condition = U>,
    T: RandomVariable,
    U: RandomVariable,
    V: RandomVariable,
{
    pub fn new(distribution: D) -> Self {
        Self {
            distribution,
            phantom: PhantomData,
        }
    }
}

impl<D, T, U, V> Distribution for TransformedDistribution<D, T, U, V>
where
    D: Distribution<Value = T, Condition = U>,
    T: RandomVariable,
    U: RandomVariable,
    V: RandomVariable,
{
    type Value = (T, V);
    type Condition = (U, V);

    fn fk(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<f64, crate::DistributionError> {
        self.distribution.fk(&x.0, &theta.0)
    }

    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, crate::DistributionError> {
        Ok((self.distribution.sample(&theta.0, rng)?, theta.1.clone()))
    }
}

impl<D, T, U, V, Rhs, TRhs> Mul<Rhs> for TransformedDistribution<D, T, U, V>
where
    D: Distribution<Value = T, Condition = U>,
    T: RandomVariable,
    U: RandomVariable,
    V: RandomVariable,
    Rhs: Distribution<Value = TRhs, Condition = (U, V)>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, (T, V), TRhs, (U, V)>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<D, T, U, V, Rhs, URhs> BitAnd<Rhs> for TransformedDistribution<D, T, U, V>
where
    D: Distribution<Value = T, Condition = U>,
    T: RandomVariable,
    U: RandomVariable,
    V: RandomVariable,
    Rhs: Distribution<Value = (U, V), Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, (T, V), (U, V), URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}
