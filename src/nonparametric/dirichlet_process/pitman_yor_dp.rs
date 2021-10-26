use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use std::{error::Error, ops::BitAnd, ops::Mul};

/// # ChineseRestaurantDP
#[derive(Clone, Debug)]
pub struct PitmanYorDP;

#[derive(thiserror::Error, Debug)]
pub enum PitmanYorDPError {
    #[error("'α' must be positibe")]
    AlphaMustBePositive,
    #[error("'d' must be greater than or equal to 0 and less than 1")]
    DMustBeGTE0AndLT1,
    #[error("Unknown error")]
    Unknown,
}

impl PitmanYorDP {
    pub fn gibbs_sampler<D, G0, T, U>(&self, params: &[U])
    where
        D: Distribution<T = T, U = U>,
        G0: Distribution<T = U, U = ()>,
        T: RandomVariable,
        U: RandomVariable,
    {
    }
}

impl Distribution for PitmanYorDP {
    type T = usize;
    type U = PitmanYorDPParams;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let alpha = theta.alpha();
        let n = theta.data_len();
        let k = *x;

        let n_vec = theta.clusters();
        let max_k = n_vec.len();

        if k <= max_k {
            Ok(n_vec[k] as f64 / (n as f64 + alpha))
        } else {
            Ok(alpha / (n as f64 + alpha))
        }
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, DistributionError> {
        let alpha = theta.alpha();
        let n = theta.data_len();

        let n_vec = theta.clusters();
        let max_k = n_vec.len();

        let p = rng.gen_range(0.0..1.0);
        let mut p_sum = 0.0;

        for k in 0..max_k {
            p_sum += n_vec[k] as f64 / (n as f64 + alpha);
            if p < p_sum {
                return Ok(k);
            }
        }

        Ok(max_k)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PitmanYorDPParams {
    alpha: f64,
    d: f64,
    z: Vec<usize>,
}

impl PitmanYorDPParams {
    /// - `alpha`: A strength parameter.
    /// - `d`: 0 ≦ d < 1. If it is zero, Pitman-Yor process means Chinese restaurant process.
    /// - `z`: `z[i]` means the index of clusters which the `i`th data belongs to.
    pub fn new(alpha: f64, d: f64, z: Vec<usize>) -> Result<Self, Box<dyn Error>> {
        if alpha <= 0.0 {
            return Err(PitmanYorDPError::AlphaMustBePositive.into());
        }
        if d < 0.0 || 1.0 <= d {
            return Err(PitmanYorDPError::DMustBeGTE0AndLT1.into());
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
        self.z.iter().fold(0usize, |max, &zi| zi.max(max))
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

impl<Rhs, TRhs> Mul<Rhs> for PitmanYorDP
where
    Rhs: Distribution<T = TRhs, U = PitmanYorDPParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, usize, TRhs, PitmanYorDPParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for PitmanYorDP
where
    Rhs: Distribution<T = PitmanYorDPParams, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, usize, PitmanYorDPParams, URhs>;

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
