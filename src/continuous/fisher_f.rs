use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use rand_distr::FisherF as RandFisherF;
use special::Beta;
use std::{ops::BitAnd, ops::Mul};

/// # FisherF
/// ![tex](https://latex.codecogs.com/svg.latex?\mathcal%7BN%7D%28\mu%2C%20\sigma%5E2%29)
#[derive(Clone, Debug)]
pub struct FisherF;

#[derive(thiserror::Error, Debug)]
pub enum FisherFError {
  #[error("'m' must be positibe")]
  MMustBePositive,
  #[error("'n' must be positibe")]
  NMustBePositive,
}

impl Distribution for FisherF {
  type T = f64;
  type U = FisherFParams;

  fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
    let m = theta.m();
    let n = theta.n();

    Ok(
      (((m * x).powf(m) * n.powf(n)) / ((m * x + n).powf(m + n))).sqrt()
        / (x * Beta::ln_beta(m / 2.0, n / 2.0).exp()),
    )
  }

  fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, DistributionError> {
    let m = theta.m();
    let n = theta.n();

    let fisher_f = match RandFisherF::new(m, n) {
      Ok(v) => Ok(v),
      Err(e) => Err(DistributionError::Others(e.into())),
    }?;

    Ok(rng.sample(fisher_f))
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FisherFParams {
  m: f64,
  n: f64,
}

impl FisherFParams {
  pub fn new(m: f64, n: f64) -> Result<Self, DistributionError> {
    if m <= 0.0 {
      return Err(DistributionError::InvalidParameters(
        FisherFError::MMustBePositive.into(),
      ));
    }
    if n <= 0.0 {
      return Err(DistributionError::InvalidParameters(
        FisherFError::NMustBePositive.into(),
      ));
    }

    Ok(Self { m, n })
  }

  pub fn m(&self) -> f64 {
    self.m
  }

  pub fn n(&self) -> f64 {
    self.n
  }
}

impl<Rhs, TRhs> Mul<Rhs> for FisherF
where
  Rhs: Distribution<T = TRhs, U = FisherFParams>,
  TRhs: RandomVariable,
{
  type Output = IndependentJoint<Self, Rhs, f64, TRhs, FisherFParams>;

  fn mul(self, rhs: Rhs) -> Self::Output {
    IndependentJoint::new(self, rhs)
  }
}

impl<Rhs, URhs> BitAnd<Rhs> for FisherF
where
  Rhs: Distribution<T = FisherFParams, U = URhs>,
  URhs: RandomVariable,
{
  type Output = DependentJoint<Self, Rhs, f64, FisherFParams, URhs>;

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
