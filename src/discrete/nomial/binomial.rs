use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
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
    type T = u64;
    type U = BinomialParams;

    fn fk(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let n = theta.n();
        let p = theta.p();

        Ok(binomial(n, *x) as f64 * p.powi(*x as i32) * (1.0 - p).powi((n - x) as i32))
    }

    fn sample(&self, theta: &Self::U, rng: &mut dyn RngCore) -> Result<Self::T, DistributionError> {
        let n = theta.n();
        let p = theta.p();

        let binominal = match RandBinominal::new(n, p) {
            Ok(v) => Ok(v),
            Err(e) => Err(DistributionError::Others(e.into())),
        }?;

        Ok(rng.sample(binominal))
    }
}

impl DiscreteDistribution for Binomial {}

#[derive(Clone, Debug, PartialEq)]
pub struct BinomialParams {
    n: u64,
    p: f64,
}

impl BinomialParams {
    pub fn new(n: u64, p: f64) -> Result<Self, DistributionError> {
        if p < 0.0 || 1.0 < p {
            return Err(DistributionError::InvalidParameters(
                BinominalError::PMustBeProbability.into(),
            ));
        }

        Ok(Self { n, p })
    }

    pub fn n(&self) -> u64 {
        self.n
    }

    pub fn p(&self) -> f64 {
        self.p
    }
}

impl<Rhs, TRhs> Mul<Rhs> for Binomial
where
    Rhs: Distribution<T = TRhs, U = BinomialParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, u64, TRhs, BinomialParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for Binomial
where
    Rhs: Distribution<T = BinomialParams, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, u64, BinomialParams, URhs>;

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
