use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use crate::{DistributionError, SamplableDistribution};
use rand::prelude::*;
use rand_distr::FisherF as RandFisherF;
use std::{ops::BitAnd, ops::Mul};

pub mod params;

pub use params::*;

/// Fisher-F distribution
/// TODO: Delete `special` package
#[derive(Clone, Debug)]
pub struct FisherF;

#[derive(thiserror::Error, Debug)]
pub enum FisherFError {
    #[error("'m' must be positibe")]
    MMustBePositive,
    #[error("'n' must be positibe")]
    NMustBePositive,
}

impl Distribution for FisherF {
    type Value = f64;
    type Condition = FisherFParams;

    fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let m = theta.m();
        let n = theta.n();

        Ok((((m * x).powf(m) * n.powf(n)) / ((m * x + n).powf(m + n))).sqrt())
    }
}

impl<Rhs, TRhs> Mul<Rhs> for FisherF
where
    Rhs: Distribution<Value = TRhs, Condition = FisherFParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, f64, TRhs, FisherFParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for FisherF
where
    Rhs: Distribution<Value = FisherFParams, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, f64, FisherFParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

impl SamplableDistribution for FisherF {
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let m = theta.m();
        let n = theta.n();

        let fisher_f = match RandFisherF::new(m, n) {
            Ok(v) => Ok(v),
            Err(e) => Err(DistributionError::Others(e.into())),
        }?;

        Ok(rng.sample(fisher_f))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
