use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use crate::{DistributionError, SampleableDistribution};
use rand::prelude::*;
use std::marker::PhantomData;
use std::{
    fmt::Debug,
    ops::{BitAnd, Mul},
};

#[derive(Clone)]
pub struct InstantDistribution<T, U, FF, FS>
where
    T: RandomVariable,
    U: RandomVariable,
    FF: Fn(&T, &U) -> Result<f64, DistributionError> + Clone + Send + Sync,
    FS: Fn(&U, &mut dyn RngCore) -> Result<T, DistributionError> + Clone + Send + Sync,
{
    fk: FF,
    sample: FS,
    phantom: PhantomData<(T, U)>,
}

impl<T, U, FF, FS> InstantDistribution<T, U, FF, FS>
where
    T: RandomVariable,
    U: RandomVariable,
    FF: Fn(&T, &U) -> Result<f64, DistributionError> + Clone + Send + Sync,
    FS: Fn(&U, &mut dyn RngCore) -> Result<T, DistributionError> + Clone + Send + Sync,
{
    pub fn new(fk: FF, sample: FS) -> Self {
        Self {
            fk,
            sample,
            phantom: PhantomData,
        }
    }
}

impl<T, U, FF, FS> Debug for InstantDistribution<T, U, FF, FS>
where
    T: RandomVariable,
    U: RandomVariable,
    FF: Fn(&T, &U) -> Result<f64, DistributionError> + Clone + Send + Sync,
    FS: Fn(&U, &mut dyn RngCore) -> Result<T, DistributionError> + Clone + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Instant")
    }
}

impl<T, U, FF, FS> Distribution for InstantDistribution<T, U, FF, FS>
where
    T: RandomVariable,
    U: RandomVariable,
    FF: Fn(&T, &U) -> Result<f64, DistributionError> + Clone + Send + Sync,
    FS: Fn(&U, &mut dyn RngCore) -> Result<T, DistributionError> + Clone + Send + Sync,
{
    type Value = T;
    type Condition = U;

    fn fk(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        (self.fk)(x, theta)
    }
}

impl<T, U, Rhs, TRhs, FF, FS> Mul<Rhs> for InstantDistribution<T, U, FF, FS>
where
    T: RandomVariable,
    U: RandomVariable,
    Rhs: Distribution<Value = TRhs, Condition = U>,
    TRhs: RandomVariable,
    FF: Fn(&T, &U) -> Result<f64, DistributionError> + Clone + Send + Sync,
    FS: Fn(&U, &mut dyn RngCore) -> Result<T, DistributionError> + Clone + Send + Sync,
{
    type Output = IndependentJoint<Self, Rhs, T, TRhs, U>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<T, U, Rhs, URhs, FF, FS> BitAnd<Rhs> for InstantDistribution<T, U, FF, FS>
where
    T: RandomVariable,
    U: RandomVariable,
    Rhs: Distribution<Value = U, Condition = URhs>,
    URhs: RandomVariable,
    FF: Fn(&T, &U) -> Result<f64, DistributionError> + Clone + Send + Sync,
    FS: Fn(&U, &mut dyn RngCore) -> Result<T, DistributionError> + Clone + Send + Sync,
{
    type Output = DependentJoint<Self, Rhs, T, U, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

impl<T, U, FF, FS> SampleableDistribution for InstantDistribution<T, U, FF, FS>
where
    T: RandomVariable,
    U: RandomVariable,
    FF: Fn(&T, &U) -> Result<f64, DistributionError> + Clone + Send + Sync,
    FS: Fn(&U, &mut dyn RngCore) -> Result<T, DistributionError> + Clone + Send + Sync,
{
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        (self.sample)(theta, rng)
    }
}
