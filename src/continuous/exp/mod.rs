use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use rand_distr::Exp as RandExp;
use std::{ops::BitAnd, ops::Mul};

/// Exponential distribution
#[derive(Clone, Debug)]
pub struct Exp;

#[derive(thiserror::Error, Debug)]
pub enum ExpError {
    #[error("Lambda must be positive")]
    LambdaMustBePositive,
}

impl Distribution for Exp {
    type T = f64;
    type U = ExpParams;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let lambda = theta.lambda();

        Ok(lambda * (-lambda * x).exp())
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, DistributionError> {
        let lambda = theta.lambda();

        let exp = match RandExp::new(lambda) {
            Ok(v) => Ok(v),
            Err(e) => Err(DistributionError::Others(e.into())),
        }?;

        Ok(rng.sample(exp))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpParams {
    lambda: f64,
}

impl ExpParams {
    pub fn new(lambda: f64) -> Result<Self, DistributionError> {
        if lambda <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                ExpError::LambdaMustBePositive.into(),
            ));
        }

        Ok(Self { lambda })
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl<Rhs, TRhs> Mul<Rhs> for Exp
where
    Rhs: Distribution<T = TRhs, U = ExpParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, f64, TRhs, ExpParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for Exp
where
    Rhs: Distribution<T = ExpParams, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, f64, ExpParams, URhs>;

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
