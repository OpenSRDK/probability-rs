use crate::{
  DependentJoint, Distribution, IndependentJoint, InverseWishart, InverseWishartParams,
  MultivariateNormal, MultivariateNormalParams, RandomVariable,
};
use opensrdk_linear_algebra::*;
use rand::prelude::*;
use std::{error::Error, ops::BitAnd, ops::Mul};

/// # NormalInverseWishart
#[derive(Clone, Debug)]
pub struct NormalInverseWishart;

#[derive(thiserror::Error, Debug)]
pub enum NormalInverseWishartError {
  #[error("Dimension mismatch")]
  DimensionMismatch,
  #[error("'λ' must be positive")]
  LambdaMustBePositive,
  #[error("'ν' must be >= dimension")]
  NuMustBeGTEDimension,
  #[error("Unknown error")]
  Unknown,
}

impl Distribution for NormalInverseWishart {
  type T = MultivariateNormalParams;
  type U = NormalInverseWishartParams;

  fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, Box<dyn Error>> {
    let mu0 = theta.mu0().clone();
    let lambda = theta.lambda();
    let lpsi = theta.lpsi().clone();
    let nu = theta.nu();

    let mu = x.mu();
    let lsigma = x.lsigma();

    let n = MultivariateNormal;
    let w_inv = InverseWishart;

    Ok(
      n.p(
        mu,
        &MultivariateNormalParams::new(mu0, (1.0 / lambda) * lsigma.clone())?,
      )? * w_inv.p(lsigma, &InverseWishartParams::new(lpsi, nu)?)?,
    )
  }

  fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, Box<dyn Error>> {
    let mu0 = theta.mu0().clone();
    let lambda = theta.lambda();
    let lpsi = theta.lpsi().clone();
    let nu = theta.nu();

    let p = MultivariateNormal;
    let winv = InverseWishart;

    let lsigma = winv.sample(&InverseWishartParams::new(lpsi, nu)?, rng)?;
    let mu = p.sample(
      &MultivariateNormalParams::new(mu0, (1.0 / lambda).sqrt() * lsigma.clone())?,
      rng,
    )?;

    Ok(MultivariateNormalParams::new(mu, lsigma)?)
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NormalInverseWishartParams {
  mu0: Vec<f64>,
  lambda: f64,
  lpsi: Matrix,
  nu: f64,
}

impl NormalInverseWishartParams {
  pub fn new(mu0: Vec<f64>, lambda: f64, lpsi: Matrix, nu: f64) -> Result<Self, Box<dyn Error>> {
    let n = mu0.len();
    if n != lpsi.rows() || n != lpsi.cols() {
      return Err(NormalInverseWishartError::DimensionMismatch.into());
    }
    if lambda <= 0.0 {
      return Err(NormalInverseWishartError::DimensionMismatch.into());
    }
    if nu <= n as f64 - 1.0 {
      return Err(NormalInverseWishartError::NuMustBeGTEDimension.into());
    }

    Ok(Self {
      mu0,
      lambda,
      lpsi,
      nu,
    })
  }

  pub fn mu0(&self) -> &Vec<f64> {
    &self.mu0
  }

  pub fn lambda(&self) -> f64 {
    self.lambda
  }

  pub fn lpsi(&self) -> &Matrix {
    &self.lpsi
  }

  pub fn nu(&self) -> f64 {
    self.nu
  }
}

impl<Rhs, TRhs> Mul<Rhs> for NormalInverseWishart
where
  Rhs: Distribution<T = TRhs, U = NormalInverseWishartParams>,
  TRhs: RandomVariable,
{
  type Output =
    IndependentJoint<Self, Rhs, MultivariateNormalParams, TRhs, NormalInverseWishartParams>;

  fn mul(self, rhs: Rhs) -> Self::Output {
    IndependentJoint::new(self, rhs)
  }
}

impl<Rhs, URhs> BitAnd<Rhs> for NormalInverseWishart
where
  Rhs: Distribution<T = NormalInverseWishartParams, U = URhs>,
  URhs: RandomVariable,
{
  type Output =
    DependentJoint<Self, Rhs, MultivariateNormalParams, NormalInverseWishartParams, URhs>;

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
