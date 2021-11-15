use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use crate::{DiscreteDistribution, DistributionError};
use num_integer::binomial;
use rand::prelude::*;
use rand_distr::Binomial as RandMultinominal;
use std::{ops::BitAnd, ops::Mul};

/// Multinominal distribution
#[derive(Clone, Debug)]
pub struct Multinominal;

#[derive(thiserror::Error, Debug)]
pub enum MultinominalError {
    #[error("'p' must be probability.")]
    PMustBeProbability,
    #[error("Unknown error")]
    Unknown,
}

impl Distribution for Multinominal {
    type Value = u64;
    type Condition = MultinominalParams;

    fn fk(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let n = theta.n();
        let p = theta.p();

        Ok(binomial(n, *x) as f64 * p.powi(*x as i32) * (1.0 - p).powi((n - x) as i32))
    }

    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let n = theta.n();
        let p = theta.p();

        let multinominal = match RandMultinominal::new(n, p) {
            Ok(v) => Ok(v),
            Err(e) => Err(DistributionError::Others(e.into())),
        }?;

        Ok(rng.sample(multinominal))
    }
}

impl DiscreteDistribution for Multinominal {}

#[derive(Clone, Debug, PartialEq)]
pub struct MultinominalParams {
    n: u64,
    p: f64,
}

impl MultinominalParams {
    pub fn new(n: u64, p: f64) -> Result<Self, DistributionError> {
        if p < 0.0 || 1.0 < p {
            return Err(DistributionError::InvalidParameters(
                MultinominalError::PMustBeProbability.into(),
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

impl<Rhs, TRhs> Mul<Rhs> for Multinominal
where
    Rhs: Distribution<Value = TRhs, Condition = MultinominalParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, u64, TRhs, MultinominalParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for Multinominal
where
    Rhs: Distribution<Value = MultinominalParams, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, u64, MultinominalParams, URhs>;

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
