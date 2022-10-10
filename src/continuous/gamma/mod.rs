// Already finished the implementation of "sampleable distribution".　The implement has commented out.

pub mod chi_squared;
pub mod params;

pub use chi_squared::*;
pub use params::*;

use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use crate::{DistributionError, SampleableDistribution};
use rand::prelude::*;
use rand_distr::Gamma as RandGamma;
use std::{ops::BitAnd, ops::Mul};

/// Gamma distribution
#[derive(Clone, Debug)]
pub struct Gamma;

#[derive(thiserror::Error, Debug)]
pub enum GammaError {
    #[error("'shape' must be positive")]
    ShapeMustBePositive,
    #[error("'scale' must be positive")]
    ScaleMustBePositive,
}

impl Distribution for Gamma {
    type Value = f64;
    type Condition = GammaParams;

    fn fk(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let shape = theta.shape();
        let scale = theta.scale();

        Ok(x.powf(shape - 1.0) * (-x / scale).exp())
    }
}

impl<Rhs, TRhs> Mul<Rhs> for Gamma
where
    Rhs: Distribution<Value = TRhs, Condition = GammaParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, f64, TRhs, GammaParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for Gamma
where
    Rhs: Distribution<Value = GammaParams, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, f64, GammaParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

impl SampleableDistribution for Gamma {
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let shape = theta.shape();
        let scale = theta.scale();

        let gamma = match RandGamma::new(shape, scale) {
            Ok(v) => Ok(v),
            Err(e) => Err(DistributionError::Others(e.into())),
        }?;

        Ok(rng.sample(gamma))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
