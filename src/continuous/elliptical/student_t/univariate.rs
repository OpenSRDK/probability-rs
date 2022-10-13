use crate::{
    ConditionDifferentiableDistribution, DependentJoint, Distribution, IndependentJoint,
    RandomVariable, SampleableDistribution, ValueDifferentiableDistribution,
};
use crate::{DistributionError, StudentTError};
use rand::prelude::*;
use rand_distr::StudentT as RandStudentT;
use special::Gamma;
use std::{ops::BitAnd, ops::Mul};

/// Student-t distribution
#[derive(Clone, Debug)]
pub struct StudentT;

impl Distribution for StudentT {
    type Value = f64;
    type Condition = StudentTParams;

    fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let nu = theta.nu();
        let mu = theta.mu();
        let sigma = theta.sigma();

        Ok((1.0 + ((x - mu) / sigma).powi(2) / nu).powf(-((nu + 1.0) / 2.0)))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StudentTParams {
    nu: f64,
    mu: f64,
    sigma: f64,
}

impl StudentTParams {
    pub fn new(nu: f64, mu: f64, sigma: f64) -> Result<Self, DistributionError> {
        if sigma <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                StudentTError::SigmaMustBePositive.into(),
            ));
        }
        Ok(Self { nu, mu, sigma })
    }

    pub fn nu(&self) -> f64 {
        self.nu
    }

    pub fn mu(&self) -> f64 {
        self.mu
    }

    pub fn sigma(&self) -> f64 {
        self.sigma
    }
}

impl<Rhs, TRhs> Mul<Rhs> for StudentT
where
    Rhs: Distribution<Value = TRhs, Condition = StudentTParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, f64, TRhs, StudentTParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for StudentT
where
    Rhs: Distribution<Value = StudentTParams, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, f64, StudentTParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

impl SampleableDistribution for StudentT {
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let nu = theta.nu();

        let student_t = match RandStudentT::new(nu) {
            Ok(v) => Ok(v),
            Err(e) => Err(DistributionError::Others(e.into())),
        }?;
        Ok(rng.sample(student_t))
    }
}

impl ValueDifferentiableDistribution for StudentT {
    fn ln_diff_value(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let mu = theta.mu();
        let x_mu = x - mu;
        let nu = theta.nu();
        let sigma = theta.sigma();
        let f_x = -(nu + 1.0) * x_mu / (nu * sigma.powi(2) + x_mu.powi(2));
        Ok(vec![f_x])
    }
}

impl ConditionDifferentiableDistribution for StudentT {
    fn ln_diff_condition(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let mu = theta.mu();
        let x_mu = x - mu;
        let sigma = theta.sigma();
        let nu = theta.nu();
        let f_mu = (nu + 1.0) * x_mu / (nu * sigma.powi(2) + x_mu.powi(2));
        let f_sigma = (nu + 1.0) * x_mu.powi(2) / (sigma * (nu * sigma.powi(2) + x_mu.powi(2)))
            - (1.0 / sigma);
        let f_nu =
            0.5 * ((nu + 1.0) / 2.0).digamma() - 0.5 * (nu / 2.0) - 1.0 / (2.0 + nu).digamma()
                + (nu + 1.0) / 2.0
                    * (1.0 + x_mu.powi(2) / (nu * sigma.powi(2))).powi(-1)
                    * x_mu.powi(2)
                    / (nu.powi(2) * sigma.powi(2))
                - 0.5 * (1.0 + x_mu / (nu * sigma.powi(2))).ln();
        Ok(vec![f_mu, f_sigma, f_nu])
    }
}

impl RandomVariable for StudentTParams {
    type RestoreInfo = ();

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        (vec![self.nu, self.mu, self.sigma], ())
    }

    fn len(&self) -> usize {
        3usize
    }

    fn restore(v: &[f64], _: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        if v.len() != 3 {
            return Err(DistributionError::InvalidRestoreVector);
        }
        Self::new(v[0], v[1], v[2])
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ConditionDifferentiableDistribution, Distribution, SampleableDistribution, StudentT,
        StudentTParams, ValueDifferentiableDistribution,
    };
    use rand::prelude::*;

    #[test]
    fn it_works() {
        let n = StudentT;
        let mut rng = StdRng::from_seed([1; 32]);

        let mu = 2.0;
        let sigma = 3.0;

        let x = n
            .sample(&StudentTParams::new(1.0, mu, sigma).unwrap(), &mut rng)
            .unwrap();

        println!("{}", x);
    }

    #[test]
    fn it_works2() {
        let n = StudentT;

        let mu = 2.0;
        let sigma = 3.0;

        let x = 0.5;

        let f = n.ln_diff_value(&x, &StudentTParams::new(1.0, mu, sigma).unwrap());
        println!("{:#?}", f);
    }

    #[test]
    fn it_works_3() {
        let n = StudentT;

        let mu = 2.0;
        let sigma = 3.0;

        let x = 0.5;

        let f = n.ln_diff_condition(&x, &StudentTParams::new(1.0, mu, sigma).unwrap());
        println!("{:#?}", f);
    }
}
