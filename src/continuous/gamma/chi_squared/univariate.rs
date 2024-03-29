// Already finished the implementation of "sampleable distribution".　The implement has commented out.

use crate::{ChiSquaredParams, DistributionError, SamplableDistribution};
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use rand_distr::ChiSquared as RandChiSquared;
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
    type Value = f64;
    type Condition = ChiSquaredParams;

    fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let k = theta.k();

        Ok(x.powf(k / 2.0 - 1.0) * (-x / 2.0).exp())
    }
}

impl<Rhs, TRhs> Mul<Rhs> for ChiSquared
where
    Rhs: Distribution<Value = TRhs, Condition = ChiSquaredParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, f64, TRhs, ChiSquaredParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for ChiSquared
where
    Rhs: Distribution<Value = ChiSquaredParams, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, f64, ChiSquaredParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

impl SamplableDistribution for ChiSquared {
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let k = theta.k();

        let chi_squared = match RandChiSquared::new(k) {
            Ok(v) => Ok(v),
            Err(e) => Err(DistributionError::Others(e.into())),
        }?;

        Ok(rng.sample(chi_squared))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
