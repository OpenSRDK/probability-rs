use crate::{
    DependentJoint, Distribution, IndependentJoint, RandomVariable, SamplableDistribution,
};
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
    U: Clone + Debug + Send + Sync,
    V: RandomVariable,
{
    distribution: D,
    phantom: PhantomData<V>,
}

impl<D, T, U, V> TransformedDistribution<D, T, U, V>
where
    D: Distribution<Value = T, Condition = U>,
    T: RandomVariable,
    U: Clone + Debug + Send + Sync,
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

    fn p_kernel(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<f64, crate::DistributionError> {
        self.distribution.p_kernel(&x.0, &theta.0)
    }
}
pub trait TransformableDistribution: Distribution + Sized {
    /// .
    ///
    /// # Examples
    ///
    /// ```
    /// // Example template not implemented for trait functions
    /// ```
    fn transform<V>(self) -> TransformedDistribution<Self, Self::Value, Self::Condition, V>
    where
        V: RandomVariable;
}

impl<D, T, U1> TransformableDistribution for D
where
    D: Distribution<Value = T, Condition = U1>,
    T: RandomVariable,
    U1: RandomVariable,
{
    fn transform<V>(self) -> TransformedDistribution<Self, Self::Value, Self::Condition, V>
    where
        V: RandomVariable,
    {
        TransformedDistribution::<Self, Self::Value, Self::Condition, V>::new(self)
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

impl<D, T, U, V> SamplableDistribution for TransformedDistribution<D, T, U, V>
where
    D: SamplableDistribution<Value = T, Condition = U>,
    T: RandomVariable,
    U: RandomVariable,
    V: RandomVariable,
{
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, crate::DistributionError> {
        Ok((self.distribution.sample(&theta.0, rng)?, theta.1.clone()))
    }
}
