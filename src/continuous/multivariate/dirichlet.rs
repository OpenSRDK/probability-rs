use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use rand_distr::Dirichlet as RandDirichlet;
use rayon::{iter::IntoParallelIterator, prelude::*};
use special::Gamma;
use std::{error::Error, ops::BitAnd, ops::Mul};

/// # Dirichlet
/// ![tex](https://latex.codecogs.com/svg.latex?\mathcal%7BN%7D%28\mu%2C%20\sigma%5E2%29)
#[derive(Clone, Debug)]
pub struct Dirichlet;

#[derive(thiserror::Error, Debug)]
pub enum DirichletError {
  #[error("Dimension mismatch")]
  DimensionMismatch,
  #[error("Length of 'α' must be >= 2")]
  AlphaLenMustBeGTE2,
  #[error("'α' must be positibe")]
  AlphaMustBePositive,
  #[error("Unknown error")]
  Unknown,
}

fn multivariate_beta(alpha: &[f64]) -> f64 {
  alpha
    .into_par_iter()
    .map(|&alphai| Gamma::gamma(alphai))
    .product::<f64>()
    / Gamma::gamma(alpha.into_par_iter().sum::<f64>())
}

impl Distribution for Dirichlet {
  type T = Vec<f64>;
  type U = DirichletParams;

  fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, Box<dyn Error>> {
    let alpha = theta.alpha();

    if x.len() != alpha.len() {
      return Err(DirichletError::DimensionMismatch.into());
    }

    Ok(
      1.0 / multivariate_beta(alpha)
        * x
          .into_par_iter()
          .zip(alpha.into_par_iter())
          .map(|(&xi, &alphai)| xi.powf(alphai - 1.0))
          .product::<f64>(),
    )
  }

  fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, Box<dyn Error>> {
    let alpha = theta.alpha();

    let dirichlet = match RandDirichlet::new(alpha) {
      Ok(n) => n,
      Err(_) => return Err(DirichletError::Unknown.into()),
    };

    Ok(rng.sample(dirichlet))
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DirichletParams {
  alpha: Vec<f64>,
}

impl DirichletParams {
  pub fn new(alpha: Vec<f64>) -> Result<Self, Box<dyn Error>> {
    if alpha.len() < 2 {
      return Err(DirichletError::AlphaLenMustBeGTE2.into());
    }
    for &alpha_i in alpha.iter() {
      if alpha_i <= 0.0 {
        return Err(DirichletError::AlphaMustBePositive.into());
      }
    }

    Ok(Self { alpha })
  }

  pub fn alpha(&self) -> &[f64] {
    &self.alpha
  }
}

impl<Rhs, TRhs> Mul<Rhs> for Dirichlet
where
  Rhs: Distribution<T = TRhs, U = DirichletParams>,
  TRhs: RandomVariable,
{
  type Output = IndependentJoint<Self, Rhs, Vec<f64>, TRhs, DirichletParams>;

  fn mul(self, rhs: Rhs) -> Self::Output {
    IndependentJoint::new(self, rhs)
  }
}

impl<Rhs, URhs> BitAnd<Rhs> for Dirichlet
where
  Rhs: Distribution<T = DirichletParams, U = URhs>,
  URhs: RandomVariable,
{
  type Output = DependentJoint<Self, Rhs, Vec<f64>, DirichletParams, URhs>;

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
