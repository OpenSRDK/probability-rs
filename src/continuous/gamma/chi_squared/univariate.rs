use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use rand_distr::ChiSquared as RandChiSquared;
use special::Gamma;
use std::{ops::BitAnd, ops::Mul};

/// Chi squared distribution
#[derive(Clone, Debug)]
pub struct ChiSquared;

#[derive(thiserror::Error, Debug)]
pub enum ChiSquaredError {
    #[error("'k' must be positibe")]
    KMustBePositive,
}

impl Distribution for ChiSquared {
    type T = f64;
    type U = ChiSquaredParams;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let k = theta.k();

        Ok((1.0 / (2f64.powf(k / 2.0) * Gamma::gamma(k / 2.0)))
            * (x.powf(k / 2.0 - 1.0) * (-x / 2.0).exp()))
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, DistributionError> {
        let k = theta.k();

        let chi_squared = match RandChiSquared::new(k) {
            Ok(v) => Ok(v),
            Err(e) => Err(DistributionError::Others(e.into())),
        }?;

        Ok(rng.sample(chi_squared))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChiSquaredParams {
    k: f64,
}

impl ChiSquaredParams {
    pub fn new(k: f64) -> Result<Self, DistributionError> {
        if k <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                ChiSquaredError::KMustBePositive.into(),
            ));
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
