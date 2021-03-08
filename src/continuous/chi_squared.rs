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
    #[error("'k' must be positibe")]
    KMustBePositive,
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

        let chi_squared = match RandChiSquared::new(k) {
            Ok(n) => n,
            Err(_) => return Err(ChiSquaredError::Unknown.into()),
        };

        Ok(rng.sample(chi_squared))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChiSquaredParams {
    k: f64,
}

impl ChiSquaredParams {
    pub fn new(k: f64) -> Result<Self, Box<dyn Error>> {
        if k <= 0.0 {
            return Err(ChiSquaredError::KMustBePositive.into());
        }

        Ok(Self { k })
    }

    pub fn k(&self) -> f64 {
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
