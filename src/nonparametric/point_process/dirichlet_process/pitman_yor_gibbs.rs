use super::PitmanYorProcessParams;
use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use std::{ops::BitAnd, ops::Mul};

/// # Pitman-Yor dirichlet process
#[derive(Clone, Debug)]
pub struct PitmanYorGibbs<G0, T>
where
    G0: Distribution<T = T, U = ()>,
    T: RandomVariable,
{
    params: PitmanYorProcessParams<G0, T>,
}

impl<G0, T> PitmanYorGibbs<G0, T>
where
    G0: Distribution<T = T, U = ()>,
    T: RandomVariable,
{
    pub fn new(params: PitmanYorProcessParams<G0, T>) -> Self {
        Self { params }
    }
}

impl<G0, T> Distribution for PitmanYorGibbs<G0, T>
where
    G0: Distribution<T = T, U = ()>,
    T: RandomVariable,
{
    type T = usize;
    type U = Vec<usize>;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
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

impl<G0, T, Rhs, TRhs> Mul<Rhs> for PitmanYorGibbs<G0, T>
where
    G0: Distribution<T = T, U = ()>,
    T: RandomVariable,
    Rhs: Distribution<T = TRhs, U = Vec<usize>>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, usize, TRhs, Vec<usize>>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<G0, T, Rhs, URhs> BitAnd<Rhs> for PitmanYorGibbs<G0, T>
where
    G0: Distribution<T = T, U = ()>,
    T: RandomVariable,
    Rhs: Distribution<T = Vec<usize>, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, usize, Vec<usize>, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}
