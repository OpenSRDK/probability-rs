use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use rand_distr::ChiSquared as RandChiSquared;
use std::{error::Error, ops::BitAnd, ops::Mul};

/// # ChiSquared
/// ![tex](https://latex.codecogs.com/svg.latex?\mathcal%7BN%7D%28\mu%2C%20\sigma%5E2%29)
#[derive(Clone, Debug)]
pub struct ChiSquared;

#[derive(thiserror::Error, Debug)]
pub enum ChiSquaredError {
    #[error("Unknown error")]
    Unknown,
}

impl Distribution for ChiSquared {
    type T = f64;
    type U = ChiSquaredParams;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, Box<dyn Error>> {
        let k = theta.k();

        Ok(todo!())
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, Box<dyn Error>> {
        let k = theta.k();

        let chisquared = match RandChiSquared::new(k as f64) {
            Ok(n) => n,
            Err(_) => return Err(ChiSquaredError::Unknown.into()),
        };

        Ok(rng.sample(chisquared))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChiSquaredParams {
    k: u64,
}

impl ChiSquaredParams {
    pub fn new(k: u64) -> Result<Self, Box<dyn Error>> {
        Ok(Self { k })
    }

    pub fn k(&self) -> u64 {
        self.k
    }
}

impl<Rhs, TRhs> Mul<Rhs> for ChiSquared
where
    Rhs: Distribution<T = TRhs, U = ChiSquaredParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, f64, TRhs, ChiSquaredParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for ChiSquared
where
    Rhs: Distribution<T = ChiSquaredParams, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, f64, ChiSquaredParams, URhs>;

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
