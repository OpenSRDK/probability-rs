use crate::TransformVec;
use crate::{
    DependentJoint, DifferentiableDistribution, Distribution, IndependentJoint, RandomVariable,
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

#[derive(Clone, Debug, PartialEq)]
pub struct NormalParams {
    mu: f64,
    sigma: f64,
}

impl NormalParams {
    pub fn new(mu: f64, sigma: f64) -> Result<Self, DistributionError> {
        if sigma <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                NormalError::SigmaMustBePositive.into(),
            ));
        }

        Ok(Self { mu, sigma })
    }

    pub fn mu(&self) -> f64 {
        self.mu
    }

    pub fn sigma(&self) -> f64 {
        self.sigma
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

impl Default for NormalParams {
    fn default() -> Self {
        Self::new(0.0, 1.0).unwrap()
    }
}

impl TransformVec for NormalParams {
    type T = ();

    fn transform_vec(self) -> (Vec<f64>, Self::T) {
        (vec![self.mu, self.sigma], ())
    }

    fn restore(v: Vec<f64>, _: Self::T) -> Self {
        Self::new(v[0], v[1]).unwrap()
    }
}

impl DifferentiableDistribution for Normal {
    fn log_diff(&self, x: &Self::Value, theta: &Self::Condition) -> Vec<f64> {
        let f = (theta.mu() - x) / theta.sigma().powi(2);
        vec![f; 1]
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
