use crate::{CauchyError, DependentJoint, Distribution, IndependentJoint, RandomVariable};
use crate::{DistributionError, StudentT, StudentTParams};
use rand::prelude::*;
use std::{ops::BitAnd, ops::Mul};

/// # Cauchy
/// ![tex](https://latex.codecogs.com/svg.latex?\mathcal%7BN%7D%28\mu%2C%20\sigma%5E2%29)
#[derive(Clone, Debug)]
pub struct Cauchy;

impl Distribution for Cauchy {
    type T = f64;
    type U = CauchyParams;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let studentt_params = StudentTParams::new(1.0, theta.mu, theta.sigma)?;

        StudentT.p(x, &studentt_params)
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, DistributionError> {
        let studentt_params = StudentTParams::new(1.0, theta.mu, theta.sigma)?;

        StudentT.sample(&studentt_params, rng)
    }
}

#[derive(Clone, Debug)]
pub struct CauchyParams {
    mu: f64,
    sigma: f64,
}

impl CauchyParams {
    pub fn new(mu: f64, sigma: f64) -> Result<Self, DistributionError> {
        if sigma <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                CauchyError::SigmaMustBePositive.into(),
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

impl<Rhs, TRhs> Mul<Rhs> for Cauchy
where
    Rhs: Distribution<T = TRhs, U = CauchyParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, f64, TRhs, CauchyParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for Cauchy
where
    Rhs: Distribution<T = CauchyParams, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, f64, CauchyParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
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
}
