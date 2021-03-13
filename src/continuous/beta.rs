use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use rand_distr::Beta as RandBeta;
use special::Beta as BetaFunc;
use std::{error::Error, ops::BitAnd, ops::Mul};

/// # Beta
/// ![tex](https://latex.codecogs.com/svg.latex?\mathcal%7BN%7D%28\mu%2C%20\sigma%5E2%29)
#[derive(Clone, Debug)]
pub struct Beta;

#[derive(thiserror::Error, Debug)]
pub enum BetaError {
  #[error("'α' must be positive")]
  AlphaMustBePositive,
  #[error("'β' must be positive")]
  BetaMustBePositive,
  #[error("Unknown error")]
  Unknown,
}

impl Distribution for Beta {
  type T = f64;
  type U = BetaParams;

  fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, Box<dyn Error>> {
    let alpha = theta.alpha();
    let beta = theta.beta();

    Ok((x.powf(alpha - 1.0) * (1.0 - x).powf(beta - 1.0)) / BetaFunc::ln_beta(alpha, beta).exp())
  }

  fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, Box<dyn Error>> {
    let alpha = theta.alpha();
    let beta = theta.beta();

    let beta = match RandBeta::new(alpha, beta) {
      Ok(n) => n,
      Err(_) => return Err(BetaError::Unknown.into()),
    };

    Ok(rng.sample(beta))
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BetaParams {
  alpha: f64,
  beta: f64,
}

impl BetaParams {
  pub fn new(alpha: f64, beta: f64) -> Result<Self, Box<dyn Error>> {
    if alpha <= 0.0 {
      return Err(BetaError::AlphaMustBePositive.into());
    }
    if alpha <= 0.0 {
      return Err(BetaError::BetaMustBePositive.into());
    }

    Ok(Self { alpha, beta })
  }

  pub fn alpha(&self) -> f64 {
    self.alpha
  }

  pub fn beta(&self) -> f64 {
    self.beta
  }
}

impl<Rhs, TRhs> Mul<Rhs> for Beta
where
  Rhs: Distribution<T = TRhs, U = BetaParams>,
  TRhs: RandomVariable,
{
  type Output = IndependentJoint<Self, Rhs, f64, TRhs, BetaParams>;

  fn mul(self, rhs: Rhs) -> Self::Output {
    IndependentJoint::new(self, rhs)
  }
}

impl<Rhs, URhs> BitAnd<Rhs> for Beta
where
  Rhs: Distribution<T = BetaParams, U = URhs>,
  URhs: RandomVariable,
{
  type Output = DependentJoint<Self, Rhs, f64, BetaParams, URhs>;

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
