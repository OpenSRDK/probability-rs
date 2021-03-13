use crate::{Distribution, RandomVariable};
use rand::prelude::*;
use rayon::prelude::*;
use std::{error::Error, f64::consts::PI};

pub trait EllipticalSliceable: RandomVariable {
  fn ellipse(self, theta: f64, nu: &Self) -> Self;
}

impl EllipticalSliceable for f64 {
  fn ellipse(self, theta: f64, nu: &Self) -> Self {
    self * theta.cos() + nu * theta.sin()
  }
}

impl EllipticalSliceable for Vec<f64> {
  fn ellipse(mut self, theta: f64, nu: &Self) -> Self {
    let cos = theta.cos();
    let sin = theta.sin();
    self
      .par_iter_mut()
      .zip(nu.par_iter())
      .for_each(|(bi, &nui)| *bi = *bi * cos + nui * sin);

    self
  }
}

/// Sample from posterior p(b|a,c) with likelihood p(a|b,c) and prior p(b|c)
pub struct EllipticalSliceSampler<'a, L, P, A, B>
where
  L: Distribution<T = A, U = B>,
  P: Distribution<T = B, U = ()>,
  A: RandomVariable,
  B: EllipticalSliceable,
{
  value: &'a A,
  likelihood: &'a L,
  prior: &'a P,
}

impl<'a, L, P, A, B> EllipticalSliceSampler<'a, L, P, A, B>
where
  L: Distribution<T = A, U = B>,
  P: Distribution<T = B, U = ()>,
  A: RandomVariable,
  B: EllipticalSliceable,
{
  pub fn new(value: &'a A, likelihood: &'a L, prior: &'a P) -> Self {
    Self {
      value,
      likelihood,
      prior,
    }
  }

  pub fn sample(&self, rng: &mut StdRng) -> Result<B, Box<dyn Error>> {
    let nu = self.prior.sample(&(), rng)?;

    let mut b = self.prior.sample(&(), rng)?;

    let rho = self.likelihood.p(self.value, &b)? * rng.gen_range(0.0..1.0);
    let mut theta = rng.gen_range(0.0..2.0 * PI);

    let mut start = theta - 2.0 * PI;
    let mut end = theta;

    loop {
      b = b.ellipse(theta, &nu);

      if rho < self.likelihood.p(self.value, &b)? {
        break;
      }

      if 0.0 < theta {
        end = 0.0;
      } else {
        start = 0.0;
      }
      theta = rng.gen_range(start..end);
    }

    Ok(b)
  }
}
