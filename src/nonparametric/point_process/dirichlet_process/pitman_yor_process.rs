use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use std::{ops::BitAnd, ops::Mul};

use super::DirichletProcessParams;

/// # Pitman-Yor process
#[derive(Clone, Debug)]
pub struct PitmanYorProcess;

#[derive(thiserror::Error, Debug)]
pub enum PitmanYorProcessError {
    #[error("'d' must be greater than or equal to 0 and less than 1")]
    DMustBeGTE0AndLT1,
    #[error("Unknown error")]
    Unknown,
}

impl Distribution for PitmanYorProcess {
    type T = usize;
    type U = PitmanYorProcessParams;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let alpha = theta.alpha();
        let n = theta.data_len();
        let k = *x;

        let n_vec = theta.clusters();
        let max_k = n_vec.len();

        if k < max_k {
            Ok((n_vec[k] as f64 - theta.d) / (n as f64 + alpha))
        } else {
            Ok((alpha + theta.d) / (n as f64 + alpha))
        }
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, DistributionError> {
        let n_vec = theta.clusters();
        let max_k = n_vec.len();

        let p = rng.gen_range(0.0..1.0);
        let mut p_sum = 0.0;

        for k in 0..max_k {
            p_sum += self.p(&k, theta)?;
            if p < p_sum {
                return Ok(k);
            }
        }

        Ok(max_k)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PitmanYorProcessParams<D, T>
where
    D: Distribution<T = T, U = ()>,
    T: RandomVariable,
{
    dirichlet: DirichletProcessParams<D, T>,
    d: f64,
    z: Vec<usize>,
}

impl PitmanYorProcessParams {
    /// - `alpha`: A strength parameter.
    /// - `d`: 0 ≦ d < 1. If it is zero, Pitman-Yor process means Chinese restaurant process.
    /// - `z`: `z[i]` means the index of clusters which the `i`th data belongs to.
    /// - `theta`: `theta[j]` means the parameters of the `j`th cluster.
    pub fn new(alpha: f64, d: f64, z: Vec<usize>) -> Result<Self, DistributionError> {
        if alpha <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                PitmanYorProcessError::AlphaMustBePositive.into(),
            ));
        }
        if d < 0.0 || 1.0 <= d {
            return Err(DistributionError::InvalidParameters(
                PitmanYorProcessError::DMustBeGTE0AndLT1.into(),
            ));
        }

        Ok(Self { alpha, d, z })
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn d(&self) -> f64 {
        self.d
    }

    pub fn z(&self) -> &Vec<usize> {
        &self.z
    }

    pub fn z_mut(&mut self) -> &mut Vec<usize> {
        &mut self.z
    }

    pub fn data_len(&self) -> usize {
        self.z.len()
    }

    pub fn clusters_len(&self) -> usize {
        self.z.iter().fold(0usize, |max, &zi| zi.max(max)) + 1
    }

    pub fn clusters(&self) -> Vec<usize> {
        let clusters_len = self.clusters_len();
        self.z
            .iter()
            .fold(vec![0usize; clusters_len], |mut n_vec, &zi| {
                n_vec[zi] += 1;
                n_vec
            })
    }
}

impl<Rhs, TRhs> Mul<Rhs> for PitmanYorProcess
where
    Rhs: Distribution<T = TRhs, U = PitmanYorProcessParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, usize, TRhs, PitmanYorProcessParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for PitmanYorProcess
where
    Rhs: Distribution<T = PitmanYorProcessParams, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, usize, PitmanYorProcessParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
