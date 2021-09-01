pub mod chi_squared;

pub use chi_squared::*;

use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use rand_distr::Gamma as RandGamma;
use special::Gamma as GammaFunc;
use std::{ops::BitAnd, ops::Mul};

/// # Gamma
/// ![tex](https://latex.codecogs.com/svg.latex?\mathcal%7BN%7D%28\mu%2C%20\sigma%5E2%29)
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
    type T = f64;
    type U = GammaParams;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let shape = theta.shape();
        let scale = theta.scale();

        Ok((1.0 / GammaFunc::gamma(shape) * scale.powf(shape))
            * x.powf(shape - 1.0)
            * (-x / scale).exp())
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, DistributionError> {
        let shape = theta.shape();
        let scale = theta.scale();

        let gamma = match RandGamma::new(shape, scale) {
            Ok(v) => Ok(v),
            Err(e) => Err(DistributionError::Others(e.into())),
        }?;

        Ok(rng.sample(gamma))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GammaParams {
    shape: f64,
    scale: f64,
}

impl GammaParams {
    pub fn new(shape: f64, scale: f64) -> Result<Self, DistributionError> {
        if shape <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                GammaError::ShapeMustBePositive.into(),
            ));
        }
        if scale <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                GammaError::ScaleMustBePositive.into(),
            ));
        }

        Ok(Self { shape, scale })
    }

    pub fn shape(&self) -> f64 {
        self.shape
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl<Rhs, TRhs> Mul<Rhs> for Gamma
where
    Rhs: Distribution<T = TRhs, U = GammaParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, f64, TRhs, GammaParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for Gamma
where
    Rhs: Distribution<T = GammaParams, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, f64, GammaParams, URhs>;

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
