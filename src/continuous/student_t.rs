use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use rand_distr::StudentT as RandStudentT;
use special::Gamma;
use std::f64::consts::PI;
use std::{error::Error, ops::BitAnd, ops::Mul};

/// # StudentT
/// ![tex](https://latex.codecogs.com/svg.latex?\mathcal%7BN%7D%28\mu%2C%20\sigma%5E2%29)
#[derive(Clone, Debug)]
pub struct StudentT;

#[derive(thiserror::Error, Debug)]
pub enum StudentTError {
  #[error("Unknown error")]
  Unknown,
}

impl Distribution for StudentT {
  type T = f64;
  type U = StudentTParams;

  fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, Box<dyn Error>> {
    let nu = theta.nu();

    Ok(
      (Gamma::gamma((nu + 1.0) / 2.0) / ((nu * PI).sqrt() * Gamma::gamma(nu / 2.0)))
        * (1.0 + x.powi(2) / nu).powf(-((nu + 1.0) / 2.0)),
    )
  }

  fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, Box<dyn Error>> {
    let nu = theta.nu();

    let student_t = match RandStudentT::new(nu) {
      Ok(n) => n,
      Err(_) => return Err(StudentTError::Unknown.into()),
    };

    Ok(rng.sample(student_t))
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StudentTParams {
  nu: f64,
}

impl StudentTParams {
  pub fn new(nu: f64) -> Result<Self, Box<dyn Error>> {
    Ok(Self { nu })
  }

  pub fn nu(&self) -> f64 {
    self.nu
  }
}

impl<Rhs, TRhs> Mul<Rhs> for StudentT
where
  Rhs: Distribution<T = TRhs, U = StudentTParams>,
  TRhs: RandomVariable,
{
  type Output = IndependentJoint<Self, Rhs, f64, TRhs, StudentTParams>;

  fn mul(self, rhs: Rhs) -> Self::Output {
    IndependentJoint::new(self, rhs)
  }
}

impl<Rhs, URhs> BitAnd<Rhs> for StudentT
where
  Rhs: Distribution<T = StudentTParams, U = URhs>,
  URhs: RandomVariable,
{
  type Output = DependentJoint<Self, Rhs, f64, StudentTParams, URhs>;

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
