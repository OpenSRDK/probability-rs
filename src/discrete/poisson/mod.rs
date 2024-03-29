pub mod params;

pub use params::*;

use crate::{
    ConditionDifferentiableDistribution, DependentJoint, Distribution, IndependentJoint,
    RandomVariable, SamplableDistribution,
};
use crate::{DiscreteDistribution, DistributionError};
use rand::prelude::*;
use rand_distr::Poisson as RandPoisson;
use std::{ops::BitAnd, ops::Mul};

/// Poisson
#[derive(Clone, Debug)]
pub struct Poisson;

#[derive(thiserror::Error, Debug)]
pub enum PoissonError {
    #[error("'λ' must be positive")]
    LambdaMustBePositive,
}

fn factorial(num: u64) -> u64 {
    match num {
        0 | 1 => 1,
        _ => factorial(num - 1) * num,
    }
}

impl Distribution for Poisson {
    type Value = u64;
    type Condition = PoissonParams;

    fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let lambda = theta.lambda();

        Ok(lambda.powi(*x as i32) / factorial(*x) as f64 * (-lambda).exp())
    }
}

impl DiscreteDistribution for Poisson {}

impl<Rhs, TRhs> Mul<Rhs> for Poisson
where
    Rhs: Distribution<Value = TRhs, Condition = PoissonParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, u64, TRhs, PoissonParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for Poisson
where
    Rhs: Distribution<Value = PoissonParams, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, u64, PoissonParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

impl ConditionDifferentiableDistribution for Poisson {
    fn ln_diff_condition(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let labmda = theta.lambda();
        let f_lambda = *x as f64 / labmda - 1.0;
        Ok(vec![f_lambda])
    }
}

impl SamplableDistribution for Poisson {
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let lambda = theta.lambda();

        let poisson = match RandPoisson::new(lambda) {
            Ok(v) => Ok(v),
            Err(e) => Err(DistributionError::Others(e.into())),
        }?;

        Ok(rng.sample(poisson) as u64)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
