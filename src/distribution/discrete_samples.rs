use crate::{Categorical, CategoricalParams, DistributionError};
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use opensrdk_linear_algebra::*;
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::Hash;
use std::{
    fmt::Debug,
    ops::{BitAnd, Mul},
};

#[derive(Clone, Debug)]
pub struct DiscreteSamplesDistribution<T>
where
    T: RandomVariable + Eq + Hash,
{
    n: usize,
    n_map: HashMap<T, usize>,
}

#[derive(thiserror::Error, Debug)]
pub enum DiscreteSamplesError {
    #[error("Samples are empty")]
    SamplesAreEmpty,
    #[error("TransformVec info mismatch")]
    TransformVecInfoMismatch,
}

impl<T> DiscreteSamplesDistribution<T>
where
    T: RandomVariable + Eq + Hash,
{
    pub fn new(samples: Vec<T>) -> Self {
        let n = samples.len();
        let mut n_map = HashMap::new();
        for sample in samples {
            *n_map.entry(sample).or_insert(0) += 1;
        }

        Self { n, n_map }
    }

    pub fn push(&mut self, v: T) {
        self.n += 1;
        *self.n_map.entry(v).or_insert(0) += 1;
    }

    pub fn mode<'a>(&'a self) -> Result<&'a T, DistributionError> {
        if self.n == 0 {
            return Err(DistributionError::InvalidParameters(
                DiscreteSamplesError::SamplesAreEmpty.into(),
            ));
        }
        Ok(self
            .n_map
            .par_iter()
            .max_by_key(|&(_, &count)| count)
            .map(|(val, _)| val)
            .unwrap_or(
                self.n_map
                    .iter()
                    .take(1)
                    .map(|(k, _)| k)
                    .collect::<Vec<_>>()[0],
            ))
    }
}

impl<T> DiscreteSamplesDistribution<T>
where
    T: RandomVariable + Eq + Hash,
{
    pub fn mean(&self) -> Result<T, DistributionError> {
        let n = self.n;
        if n == 0 {
            return Err(DistributionError::InvalidParameters(
                DiscreteSamplesError::SamplesAreEmpty.into(),
            ));
        }
        let vec = self.n_map.iter().collect::<Vec<_>>();
        let (sum, info) = vec[0].0.clone().transform_vec();
        let mut sum = sum.col_mat();
        for i in 1..n {
            let (v, info_i) = vec[i].0.clone().transform_vec();
            if info != info_i {
                return Err(DistributionError::Others(
                    DiscreteSamplesError::TransformVecInfoMismatch.into(),
                ));
            }
            sum = sum + (*vec[i].1 as f64) * v.col_mat();
        }

        T::restore(sum.elems(), &info)
    }
}

impl<T> Distribution for DiscreteSamplesDistribution<T>
where
    T: RandomVariable + Eq + Hash,
{
    type Value = T;
    type Condition = ();

    fn fk(&self, x: &Self::Value, _: &Self::Condition) -> Result<f64, DistributionError> {
        Ok(*self.n_map.get(x).unwrap_or(&0) as f64 / self.n as f64)
    }

    fn sample(
        &self,
        _theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let keys = self.n_map.keys().collect::<Vec<_>>();
        let pi = keys
            .iter()
            .map(|&k| *self.n_map.get(k).unwrap())
            .map(|ni| ni as f64 / self.n as f64)
            .collect();
        let params = CategoricalParams::new(pi)?;
        let sampled = Categorical.sample(&params, rng)?;

        Ok(keys[sampled].clone())
    }
}

impl<T, Rhs, TRhs> Mul<Rhs> for DiscreteSamplesDistribution<T>
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

impl<T, Rhs, URhs> BitAnd<Rhs> for DiscreteSamplesDistribution<T>
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
