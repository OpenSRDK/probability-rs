use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use std::{error::Error, ops::BitAnd, ops::Mul};

/// # ChineseRestaurantDP
/// ![tex](https://latex.codecogs.com/svg.latex?\mathcal%7BN%7D%28\mu%2C%20\sigma%5E2%29)
#[derive(Clone, Debug)]
pub struct ChineseRestaurantDP;

#[derive(thiserror::Error, Debug)]
pub enum ChineseRestaurantDPError {
  #[error("'α' must be positibe")]
  AlphaMustBePositive,
  #[error("Unknown error")]
  Unknown,
}

impl Distribution for ChineseRestaurantDP {
  type T = u64;
  type U = ChineseRestaurantDPParams;

  fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
    let i = theta.i();
    let alpha = theta.alpha();
    let z = theta.z();
    let k = *x as usize;

    let max_k = z.iter().fold(0u64, |max, &zi| zi.max(max)) as usize;
    let n = z.iter().fold(vec![0u32; max_k], |mut n, &zi| {
      n[zi as usize] += 1;
      n
    });

    if k <= max_k {
      Ok(n[k] as f64 / (i as f64 + alpha))
    } else {
      Ok(alpha / (i as f64 + alpha))
    }
  }

  fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, DistributionError> {
    let i = theta.i();
    let alpha = theta.alpha();
    let z = theta.z();

    let max_k = z.iter().fold(0u64, |max, &zi| zi.max(max)) as usize;
    let n = z.iter().fold(vec![0u32; max_k], |mut n, &zi| {
      n[zi as usize] += 1;
      n
    });

    let p = rng.gen_range(0.0..1.0);
    let mut p_sum = 0.0;

    for k in 0..max_k {
      p_sum += n[k] as f64 / (i as f64 + alpha);
      if p < p_sum {
        return Ok(k as u64);
      }
    }

    Ok(max_k as u64 + 1)
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChineseRestaurantDPParams {
  i: usize,
  alpha: f64,
  z: Vec<u64>,
}

impl ChineseRestaurantDPParams {
  pub fn new(i: usize, alpha: f64, z: Vec<u64>) -> Result<Self, Box<dyn Error>> {
    if alpha <= 0.0 {
      return Err(ChineseRestaurantDPError::AlphaMustBePositive.into());
    }

    Ok(Self { i, alpha, z })
  }

  pub fn i(&self) -> usize {
    self.i
  }

  pub fn alpha(&self) -> f64 {
    self.alpha
  }

  pub fn z(&self) -> &[u64] {
    &self.z
  }
}

impl<Rhs, TRhs> Mul<Rhs> for ChineseRestaurantDP
where
  Rhs: Distribution<T = TRhs, U = ChineseRestaurantDPParams>,
  TRhs: RandomVariable,
{
  type Output = IndependentJoint<Self, Rhs, u64, TRhs, ChineseRestaurantDPParams>;

  fn mul(self, rhs: Rhs) -> Self::Output {
    IndependentJoint::new(self, rhs)
  }
}

impl<Rhs, URhs> BitAnd<Rhs> for ChineseRestaurantDP
where
  Rhs: Distribution<T = ChineseRestaurantDPParams, U = URhs>,
  URhs: RandomVariable,
{
  type Output = DependentJoint<Self, Rhs, u64, ChineseRestaurantDPParams, URhs>;

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
