use crate::{DependentJoint, Distribution, RandomVariable};
use rand::prelude::StdRng;
use std::{error::Error, ops::BitAnd, ops::Mul};

/// # IndependentJoint
/// ![tex](https://latex.codecogs.com/svg.latex?p%28a,b%7Cc%29%3Dp%28a%7Cc%29p%28b%7Cc%29)
#[derive(Clone, Debug)]
pub struct IndependentJoint<L, R, TL, TR, U>
where
  L: Distribution<T = TL, U = U>,
  R: Distribution<T = TR, U = U>,
  TL: RandomVariable,
  TR: RandomVariable,
  U: RandomVariable,
{
  lhs: L,
  rhs: R,
}

impl<L, R, TL, TR, U> IndependentJoint<L, R, TL, TR, U>
where
  L: Distribution<T = TL, U = U>,
  R: Distribution<T = TR, U = U>,
  TL: RandomVariable,
  TR: RandomVariable,
  U: RandomVariable,
{
  pub fn new(lhs: L, rhs: R) -> Self {
    Self { lhs, rhs }
  }
}

impl<L, R, TL, TR, U> Distribution for IndependentJoint<L, R, TL, TR, U>
where
  L: Distribution<T = TL, U = U>,
  R: Distribution<T = TR, U = U>,
  TL: RandomVariable,
  TR: RandomVariable,
  U: RandomVariable,
{
  type T = (TL, TR);
  type U = U;

  fn p(&self, x: &(TL, TR), theta: &U) -> Result<f64, Box<dyn Error>> {
    Ok(self.lhs.p(&x.0, theta)? * self.rhs.p(&x.1, theta)?)
  }

  fn sample(&self, theta: &U, rng: &mut StdRng) -> Result<(TL, TR), Box<dyn Error>> {
    Ok((self.lhs.sample(theta, rng)?, self.rhs.sample(theta, rng)?))
  }
}

impl<L, R, TL, TR, U, Rhs, TRhs> Mul<Rhs> for IndependentJoint<L, R, TL, TR, U>
where
  L: Distribution<T = TL, U = U>,
  R: Distribution<T = TR, U = U>,
  TL: RandomVariable,
  TR: RandomVariable,
  U: RandomVariable,
  Rhs: Distribution<T = TRhs, U = U>,
  TRhs: RandomVariable,
{
  type Output = IndependentJoint<Self, Rhs, (TL, TR), TRhs, U>;

  fn mul(self, rhs: Rhs) -> Self::Output {
    IndependentJoint::new(self, rhs)
  }
}

impl<L, R, TL, TR, U, Rhs, URhs> BitAnd<Rhs> for IndependentJoint<L, R, TL, TR, U>
where
  L: Distribution<T = TL, U = U>,
  R: Distribution<T = TR, U = U>,
  TL: RandomVariable,
  TR: RandomVariable,
  U: RandomVariable,
  Rhs: Distribution<T = U, U = URhs>,
  URhs: RandomVariable,
{
  type Output = DependentJoint<Self, Rhs, (TL, TR), U, URhs>;

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
    let model = Normal * Normal;

    let mut rng = StdRng::from_seed([1; 32]);

    let x = model
      .sample(&NormalParams::new(0.0, 1.0).unwrap(), &mut rng)
      .unwrap();

    println!("{:#?}", x);
  }
}
