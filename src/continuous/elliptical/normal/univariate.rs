use crate::{
    ConditionDifferentiableDistribution, DependentJoint, Distribution, IndependentJoint,
    NormalParams, RandomVariable, ValueDifferentiableDistribution,
};
use crate::{DistributionError, NormalError};
use rand::prelude::*;
use rand_distr::Normal as RandNormal;
use std::{ops::BitAnd, ops::Mul};

/// Normal distribution
#[derive(Clone, Debug)]
pub struct Normal;

impl Distribution for Normal {
    type Value = f64;
    type Condition = NormalParams;

    fn fk(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let mu = theta.mu();
        let sigma = theta.sigma();

        Ok((-(x - mu).powi(2) / (2.0 * sigma.powi(2))).exp())
    }

    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let mu = theta.mu();
        let sigma = theta.sigma();

        let normal = match RandNormal::new(mu, sigma) {
            Ok(n) => n,
            Err(_) => {
                return Err(DistributionError::InvalidParameters(
                    NormalError::SigmaMustBePositive.into(),
                ))
            }
        };

        Ok(rng.sample(normal))
    }
}

impl<Rhs, TRhs> Mul<Rhs> for Normal
where
    Rhs: Distribution<Value = TRhs, Condition = NormalParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, f64, TRhs, NormalParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for Normal
where
    Rhs: Distribution<Value = NormalParams, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, f64, NormalParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

impl ValueDifferentiableDistribution for Normal {
    fn ln_diff_value(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let f = (theta.mu() - x) / theta.sigma().powi(2);
        Ok(vec![f; 1])
    }
}

impl ConditionDifferentiableDistribution for Normal {
    fn ln_diff_condition(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let sigma = theta.sigma();
        let mu = theta.mu();
        let fk = self.fk(x, theta).unwrap();
        let f_mu = (x - mu) / sigma.powi(2) * fk;
        let f_sigma = -(x - mu).powi(2) / sigma.powi(4) * fk;
        Ok(vec![f_mu, f_sigma])
    }
}

#[cfg(test)]
mod tests {
    use crate::{Distribution, Normal, NormalParams};
    use rand::prelude::*;

    #[test]
    fn it_works() {
        let n = Normal;
        let mut rng = StdRng::from_seed([1; 32]);

        let mu = 2.0;
        let sigma = 3.0;

        let x = n
            .sample(&NormalParams::new(mu, sigma).unwrap(), &mut rng)
            .unwrap();

        println!("{}", x);
    }
}
