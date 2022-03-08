use crate::{
    ConditionDifferentiableDistribution, DependentJoint, Distribution, IndependentJoint,
    RandomVariable, ValueDifferentiableDistribution,
};
use crate::{DistributionError, StudentTError};
use rand::prelude::*;
use rand_distr::StudentT as RandStudentT;
use std::{ops::BitAnd, ops::Mul};

/// Student-t distribution
#[derive(Clone, Debug)]
pub struct StudentT;

impl Distribution for StudentT {
    type Value = f64;
    type Condition = StudentTParams;

    fn fk(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let nu = theta.nu();
        let mu = theta.mu();
        let sigma = theta.sigma();

        Ok((1.0 + ((x - mu) / sigma).powi(2) / nu).powf(-((nu + 1.0) / 2.0)))
    }

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

impl ValueDifferentiableDistribution for StudentT {
    fn ln_diff_value(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let nu = theta.nu();
        let mu = theta.mu();
        let f_x = -(nu + 1.0) / (x - mu);
        Ok(vec![f_x])
    }
}

impl ConditionDifferentiableDistribution for StudentT {
    fn ln_diff_condition(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let nu = theta.nu();
        let mu = theta.mu();
        let sigma = theta.sigma();
        let f_mu = (nu + 1.0) / (x - mu);
        let f_sigma = (nu + 1.0) / sigma;
        Ok(vec![f_mu, f_sigma])
    }
}

impl RandomVariable for StudentTParams {
    type RestoreInfo = ();

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        (vec![self.nu, self.mu, self.sigma], ())
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
    use crate::{Distribution, StudentT, StudentTParams};
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
}
