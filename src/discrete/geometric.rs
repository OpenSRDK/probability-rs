use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use rand_distr::Geometric as RandGeometric;
use std::{error::Error, ops::BitAnd, ops::Mul};

/// # Geometric
/// ![tex](https://latex.codecogs.com/svg.latex?\mathcal%7BN%7D%28\mu%2C%20\sigma%5E2%29)
#[derive(Clone, Debug)]
pub struct Geometric;

#[derive(thiserror::Error, Debug)]
pub enum GeometricError {
  #[error("'p' must be probability.")]
  PMustBeProbability,
  #[error("Unknown error")]
  Unknown,
}

impl Distribution for Geometric {
  type T = u64;
  type U = GeometricParams;

  fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, Box<dyn Error>> {
    let p = theta.p();

    Ok((1.0 - p).powi((x - 1) as i32) * p)
  }

  fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, Box<dyn Error>> {
    let p = theta.p();

    let geometric = match RandGeometric::new(p) {
      Ok(n) => n,
      Err(_) => return Err(GeometricError::Unknown.into()),
    };

    Ok(rng.sample(geometric))
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeometricParams {
  p: f64,
}

impl GeometricParams {
  pub fn new(p: f64) -> Result<Self, Box<dyn Error>> {
    if p < 0.0 || 1.0 < p {
      return Err(GeometricError::PMustBeProbability.into());
    }

    Ok(Self { p })
  }

  pub fn p(&self) -> f64 {
    self.p
  }
}

impl<Rhs, TRhs> Mul<Rhs> for Geometric
where
  Rhs: Distribution<T = TRhs, U = GeometricParams>,
  TRhs: RandomVariable,
{
  type Output = IndependentJoint<Self, Rhs, u64, TRhs, GeometricParams>;

  fn mul(self, rhs: Rhs) -> Self::Output {
    IndependentJoint::new(self, rhs)
  }
}

impl<Rhs, URhs> BitAnd<Rhs> for Geometric
where
  Rhs: Distribution<T = GeometricParams, U = URhs>,
  URhs: RandomVariable,
{
  type Output = DependentJoint<Self, Rhs, u64, GeometricParams, URhs>;

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
