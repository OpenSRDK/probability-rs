use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use rand_distr::Gamma as RandGamma;
use std::{error::Error, f64::consts::PI, ops::BitAnd, ops::Mul};

/// # Gamma
/// ![tex](https://latex.codecogs.com/svg.latex?\mathcal%7BN%7D%28\mu%2C%20\sigma%5E2%29)
#[derive(Clone, Debug)]
pub struct Gamma;

#[derive(thiserror::Error, Debug)]
pub enum GammaError {
    #[error("Unknown error")]
    Unknown,
}

impl Distribution for Gamma {
    type T = f64;
    type U = GammaParams;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, Box<dyn Error>> {
        let mu = theta.mu();
        let sigma = theta.sigma();

        Ok(1.0 / (2.0 * PI * sigma.powi(2)).sqrt()
            * (-(x - mu).powi(2) / (2.0 * sigma.powi(2))).exp())
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, Box<dyn Error>> {
        let mu = theta.mu();
        let sigma = theta.sigma();

        let gamma = match RandGamma::new(mu, sigma) {
            Ok(n) => n,
            Err(_) => return Err(GammaError::Unknown.into()),
        };

        Ok(rng.sample(gamma))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GammaParams {
    mu: f64,
    sigma: f64,
}

impl GammaParams {
    pub fn new(mu: f64, sigma: f64) -> Result<Self, Box<dyn Error>> {
        if sigma <= 0.0 {
            return Err(GammaError::Unknown.into());
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

impl<Rhs, TRhs> Mul<Rhs> for Gamma
where
    Rhs: Distribution<T = TRhs, U = GammaParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, f64, TRhs, GammaParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for Gamma
where
    Rhs: Distribution<T = GammaParams, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, f64, GammaParams, URhs>;

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
