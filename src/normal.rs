use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use rand_distr::Normal as RandNormal;
use std::{error::Error, f64::consts::PI, ops::BitAnd, ops::Mul};

#[derive(Clone, Debug)]
pub struct Normal;

#[derive(thiserror::Error, Debug)]
pub enum NormalError {
    #[error("invalid variance")]
    InvalidVariance,
}

impl Distribution for Normal {
    type T = f64;
    type U = NormalParams;

    fn p(&self, x: &f64, theta: &NormalParams) -> Result<f64, Box<dyn Error>> {
        let mu = theta.mu();
        let sigma = theta.sigma();

        Ok(1.0 / (2.0 * PI * sigma.powi(2)).sqrt()
            * (-(x - mu).powi(2) / (2.0 * sigma.powi(2))).exp())
    }

    fn sample(&self, theta: &NormalParams, rng: &mut StdRng) -> Result<f64, Box<dyn Error>> {
        let mu = theta.mu();
        let sigma = theta.sigma();

        let normal = match RandNormal::new(mu, sigma) {
            Ok(n) => n,
            Err(_) => return Err(NormalError::InvalidVariance.into()),
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
    pub fn new(mu: f64, sigma: f64) -> Result<Self, Box<dyn Error>> {
        if sigma <= 0.0 {
            return Err(NormalError::InvalidVariance.into());
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
    Rhs: Distribution<T = TRhs, U = NormalParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, f64, TRhs, NormalParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for Normal
where
    Rhs: Distribution<T = NormalParams, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, f64, NormalParams, URhs>;

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
