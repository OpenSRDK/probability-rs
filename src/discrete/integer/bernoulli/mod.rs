pub mod params;

pub use params::*;

use crate::*;
use rand::Rng;
use std::ops::{BitAnd, Mul};

#[derive(Clone, Debug)]
pub struct Bernoulli;

#[derive(thiserror::Error, Debug)]
pub enum BernoulliError {
    #[error("'p' must be probability.")]
    PMustBeProbability,
}

impl Distribution for Bernoulli {
    type Value = bool;
    type Condition = BernoulliParams;

    fn fk(&self, _x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        Ok(theta.p())
    }

    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn rand::RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let u = rng.gen_range(0.0..=1.0);
        Ok(u <= theta.p())
    }
}

impl DiscreteDistribution for Bernoulli {}

impl<Rhs, TRhs> Mul<Rhs> for Bernoulli
where
    Rhs: Distribution<Value = TRhs, Condition = BernoulliParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, bool, TRhs, BernoulliParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for Bernoulli
where
    Rhs: Distribution<Value = BernoulliParams, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, bool, BernoulliParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

impl ConditionDifferentiableDistribution for Bernoulli {
    fn ln_diff_condition(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let p = theta.p();
        // let x_f64 = if *x { 1.0 } else { 0.0 };
        // let f_p = x_f64 / p - (1.0 - x_f64) / (1.0 - p);
        let f_p = if *x { 1.0 } else { -1.0 };
        Ok(vec![f_p])
    }
}
