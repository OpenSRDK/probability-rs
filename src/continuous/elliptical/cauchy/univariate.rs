use crate::{
    CauchyParams, ConditionDifferentiableDistribution, DependentJoint, Distribution,
    IndependentJoint, RandomVariable, SampleableDistribution, ValueDifferentiableDistribution,
};
use crate::{DistributionError, StudentT, StudentTParams};
use rand::prelude::*;
use std::{ops::BitAnd, ops::Mul};

/// Cauchy distribution
#[derive(Clone, Debug)]
pub struct Cauchy;

impl Distribution for Cauchy {
    type Value = f64;
    type Condition = CauchyParams;

    fn fk(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let studentt_params = StudentTParams::new(1.0, theta.mu(), theta.sigma())?;

        StudentT.fk(x, &studentt_params)
    }
}

impl<Rhs, TRhs> Mul<Rhs> for Cauchy
where
    Rhs: Distribution<Value = TRhs, Condition = CauchyParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, f64, TRhs, CauchyParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for Cauchy
where
    Rhs: Distribution<Value = CauchyParams, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, f64, CauchyParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

impl SampleableDistribution for Cauchy {
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let studentt_params = StudentTParams::new(1.0, theta.mu(), theta.sigma())?;

        StudentT.sample(&studentt_params, rng)
    }
}

impl ValueDifferentiableDistribution for Cauchy {
    fn ln_diff_value(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let mu = theta.mu();
        let x_mu = x - mu;
        let sigma = theta.sigma();
        let f_x = -2.0 * x_mu / (sigma.powi(2) + x_mu.powi(2));
        Ok(vec![f_x])
    }
}

impl ConditionDifferentiableDistribution for Cauchy {
    fn ln_diff_condition(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let mu = theta.mu();
        let x_mu = x - mu;
        let sigma = theta.sigma();
        let f_mu = 2.0 * x_mu / (sigma.powi(2) + x_mu.powi(2));
        let f_sigma = 2.0 * x_mu.powi(2) / (sigma * (sigma.powi(2) + x_mu.powi(2))) - (1.0 / sigma);
        Ok(vec![f_mu, f_sigma])
    }
}

#[cfg(test)]
mod tests {
    use crate::distribution::Distribution;
    use crate::*;
    use rand::prelude::*;

    #[test]
    fn it_works() {
        let n = Cauchy;
        let mut rng = StdRng::from_seed([1; 32]);

        let mu = 2.0;
        let sigma = 3.0;

        let x = n
            .sample(&CauchyParams::new(mu, sigma).unwrap(), &mut rng)
            .unwrap();

        println!("{}", x);
    }
    #[test]
    fn it_works2() {
        let n = Cauchy;

        let mu = 2.0;
        let sigma = 3.0;

        let x = 0.5;

        let f = n.ln_diff_value(&x, &CauchyParams::new(mu, sigma).unwrap());
        println!("{:#?}", f);
    }

    #[test]
    fn it_works_3() {
        let n = Cauchy;

        let mu = 2.0;
        let sigma = 3.0;

        let x = 0.5;

        let f = n.ln_diff_condition(&x, &CauchyParams::new(mu, sigma).unwrap());
        println!("{:#?}", f);
    }
}
