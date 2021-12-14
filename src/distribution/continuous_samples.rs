use crate::mcmc::*;
use crate::{Categorical, CategoricalParams, DistributionError};
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use std::hash::Hash;
use std::iter::Sum;
use std::ops::Div;
use std::{
    fmt::Debug,
    ops::{BitAnd, Mul},
};

#[derive(Clone, Debug)]
pub struct ContinuousSamplesDistribution<T>
where
    T: RandomVariable,
{
    samples: Vec<T>,
}

#[derive(thiserror::Error, Debug)]
pub enum SamplesError {
    #[error("Samples are empty")]
    SamplesAreEmpty,
}

impl<T> ContinuousSamplesDistribution<T>
where
    T: RandomVariable + VectorSampleable + Sum + Div<f64, Output = T>,
{
    pub fn new(samples: Vec<T>) -> Self {
        Self { samples }
    }

    pub fn samples(&self) -> &Vec<T> {
        &self.samples
    }

    pub fn samples_mut(&mut self) -> &mut Vec<T> {
        &mut self.samples
    }
}

impl<T> Distribution for ContinuousSamplesDistribution<T>
where
    T: RandomVariable + PartialEq,
{
    type Value = T;
    type Condition = ();

    fn fk(&self, x: &Self::Value, _: &Self::Condition) -> Result<f64, DistributionError> {
        let eq_num = &self
            .samples
            .iter()
            .map(|sample| -> f64 {
                if sample == x {
                    1.0
                } else {
                    0.0
                }
            })
            .sum::<f64>();
        Ok(eq_num / self.samples.len() as f64)
    }

    fn sample(
        &self,
        _theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let pi = vec![1.0 / self.samples.len() as f64; self.samples.len()];
        let params = CategoricalParams::new(pi)?;
        let sampled = Categorical.sample(&params, rng)?;
        Ok(self.samples[sampled].clone())
    }
}

impl<T, Rhs, TRhs> Mul<Rhs> for ContinuousSamplesDistribution<T>
where
    T: RandomVariable + Eq + Hash,
    Rhs: Distribution<Value = TRhs, Condition = ()>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, T, TRhs, ()>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<T, Rhs, URhs> BitAnd<Rhs> for ContinuousSamplesDistribution<T>
where
    T: RandomVariable + Eq + Hash,
    Rhs: Distribution<Value = (), Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, T, (), URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}
