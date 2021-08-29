use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use crate::{DistributionError, NormalError};
use rand::prelude::*;
use rand_distr::Normal as RandNormal;
use std::{f64::consts::PI, ops::BitAnd, ops::Mul};

/// # Normal
/// ![tex](https://latex.codecogs.com/svg.latex?\mathcal%7BN%7D%28\mu%2C%20\sigma%5E2%29)
#[derive(Clone, Debug)]
pub struct Normal;

impl Distribution for Normal {
    type T = f64;
    type U = NormalParams;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let mu = theta.mu();
        let sigma = theta.sigma();

        Ok(1.0 / (2.0 * PI * sigma.powi(2)).sqrt()
            * (-(x - mu).powi(2) / (2.0 * sigma.powi(2))).exp())
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, DistributionError> {
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
    use crate::{Cauchy, CauchyParams, Distribution};
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
}
