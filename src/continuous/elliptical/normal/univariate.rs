use crate::{
    ConditionDifferentiableDistribution, DependentJoint, Distribution, IndependentJoint,
    NormalParams, RandomVariable, SamplableDistribution, ValueDifferentiableDistribution,
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

    fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let mu = theta.mu();
        let sigma = theta.sigma();

        Ok((-(x - mu).powi(2) / (2.0 * sigma.powi(2))).exp())
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

impl SamplableDistribution for Normal {
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

impl ValueDifferentiableDistribution for Normal {
    fn ln_diff_value(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let sigma = theta.sigma();
        let mu = theta.mu();
        let f_x = -(x - mu) / sigma.powi(2);
        Ok(vec![f_x])
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
        let f_mu = (x - mu) / sigma.powi(2);
        let f_sigma = (x - mu).powi(2) / sigma.powi(3) - 1.0 / sigma;
        Ok(vec![f_mu, f_sigma])
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ConditionDifferentiableDistribution, Distribution, Normal, NormalParams,
        SamplableDistribution, ValueDifferentiableDistribution,
    };
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
        let result = n.p_kernel(&0.5, &NormalParams::new(0.0, 1.0).unwrap());
    }

    #[test]
    fn it_works2() {
        let n = Normal;

        let mu = 2.0;
        let sigma = 3.0;

        let x = 0.5;

        let f = n.ln_diff_value(&x, &NormalParams::new(mu, sigma).unwrap());
        println!("{:#?}", f);
    }

    #[test]
    fn it_works_3() {
        let n = Normal;

        let mu = 0.0;
        let sigma = 5.0;

        let x = 1.0;

        let f = n.ln_diff_condition(&x, &NormalParams::new(mu, sigma).unwrap());
        println!("{:#?}", f);
    }
}
