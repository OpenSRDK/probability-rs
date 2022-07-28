use crate::{Categorical, CategoricalParams, DistributionError};
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use opensrdk_linear_algebra::*;
use rand::prelude::*;
use std::hash::Hash;
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
pub enum ContinuousSamplesError {
    #[error("Samples are empty")]
    SamplesAreEmpty,
    #[error("TransformVec info mismatch")]
    TransformVecInfoMismatch,
}

impl<T> ContinuousSamplesDistribution<T>
where
    T: RandomVariable,
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

    pub fn sum(&self) -> Result<T, DistributionError> {
        let n = self.samples.len();
        if n == 0 {
            return Err(DistributionError::Others(
                ContinuousSamplesError::SamplesAreEmpty.into(),
            ));
        }
        let (sum, info) = self.samples[0].clone().transform_vec();
        let mut sum = sum.col_mat();
        for i in 1..n {
            let (v, info_i) = self.samples[i].clone().transform_vec();
            if info != info_i {
                return Err(DistributionError::Others(
                    ContinuousSamplesError::TransformVecInfoMismatch.into(),
                ));
            }
            sum = sum + v.col_mat();
        }

        T::restore(sum.elems(), &info)
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
