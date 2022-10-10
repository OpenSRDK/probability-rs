pub mod params;

pub use params::*;

use crate::{
    DependentJoint, Distribution, IndependentJoint, RandomVariable, SampleableDistribution,
};
use crate::{DiscreteDistribution, DistributionError};
use num_integer::binomial;
use rand::prelude::*;
use rand_distr::Binomial as RandBinominal;
use std::{ops::BitAnd, ops::Mul};

/// Binominal distribution
#[derive(Clone, Debug)]
pub struct Binomial;

#[derive(thiserror::Error, Debug)]
pub enum BinominalError {
    #[error("'p' must be probability.")]
    PMustBeProbability,
    #[error("Unknown error")]
    Unknown,
}

impl Distribution for Binomial {
    type Value = u64;
    type Condition = BinomialParams;

    fn fk(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let n = theta.n();
        let p = theta.p();

        Ok(binomial(n, *x) as f64 * p.powi(*x as i32) * (1.0 - p).powi((n - x) as i32))
    }
}

impl DiscreteDistribution for Binomial {}

impl<Rhs, TRhs> Mul<Rhs> for Binomial
where
    Rhs: Distribution<Value = TRhs, Condition = BinomialParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, u64, TRhs, BinomialParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for Binomial
where
    Rhs: Distribution<Value = BinomialParams, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, u64, BinomialParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

impl SampleableDistribution for Binomial {
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let n = theta.n();
        let p = theta.p();

        let binominal = match RandBinominal::new(n, p) {
            Ok(v) => Ok(v),
            Err(e) => Err(DistributionError::Others(e.into())),
        }?;

        Ok(rng.sample(binominal))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
