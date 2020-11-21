use crate::{Distribution, RandomVariable};
use rand::prelude::*;
use rayon::prelude::*;
use std::{error::Error, f64::consts::PI, marker::PhantomData};

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
        self.par_iter_mut()
            .zip(nu.par_iter())
            .for_each(|(bi, &nui)| *bi = *bi * cos + nui * sin);

        self
    }
}

/// Sample from posterior p(b|a) with likelihood p(a|b) and prior p(b)
pub struct EllipticalSliceSampler<'a, L, R, T, UL, UR>
where
    L: Distribution<T = T, U = UL>,
    R: Distribution<T = UL, U = UR>,
    T: RandomVariable,
    UL: EllipticalSliceable,
    UR: RandomVariable,
{
    value: &'a T,
    likelihood: &'a L,
    prior: &'a R,
    phantom: PhantomData<(T, UL)>,
}

impl<'a, DL, DP, T, UL, UR> EllipticalSliceSampler<'a, DL, DP, T, UL, UR>
where
    DL: Distribution<T = T, U = UL>,
    DP: Distribution<T = UL, U = UR>,
    T: RandomVariable,
    UL: EllipticalSliceable,
    UR: RandomVariable,
{
    pub fn new(value: &'a T, likelihood: &'a DL, prior: &'a DP) -> Self {
        Self {
            value,
            likelihood,
            prior,
            phantom: PhantomData,
        }
    }

    pub fn sample(&self, theta: &UR, rng: &mut StdRng) -> Result<UL, Box<dyn Error>> {
        let nu = self.prior.sample(theta, rng)?;

        let mut b = self.prior.sample(theta, rng)?;

        let rho = self.likelihood.p(self.value, &b)? * rng.gen_range(0.0, 1.0);
        let mut theta = rng.gen_range(0.0, 2.0 * PI);

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
            theta = rng.gen_range(start, end);
        }

        Ok(b)
    }
}
