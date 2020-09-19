use crate::Distribution;
use rand::prelude::*;
use rayon::prelude::*;
use std::{error::Error, f64::consts::PI};

pub trait EllipticalSliceable: Clone {
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
pub struct EllipticalSliceSampler<T>
where
    T: EllipticalSliceable,
{
    likelihood: Box<dyn Fn(&T) -> Result<f64, Box<dyn Error>>>,
    prior: Box<dyn Distribution<T>>,
}

impl<T> EllipticalSliceSampler<T>
where
    T: EllipticalSliceable,
{
    pub fn new(
        likelihood: Box<dyn Fn(&T) -> Result<f64, Box<dyn Error>>>,
        prior: Box<dyn Distribution<T>>,
    ) -> Self {
        Self { likelihood, prior }
    }

    pub fn sample(&self, rng: &mut StdRng) -> Result<T, Box<dyn Error>> {
        let nu = self.prior.sample(rng)?;

        let mut b = self.prior.sample(rng)?;
        let rho = (self.likelihood)(&b)? * rng.gen_range(0.0, 1.0);
        let mut theta = rng.gen_range(0.0, 2.0 * PI);

        let mut start = theta - 2.0 * PI;
        let mut end = theta;

        loop {
            b = b.ellipse(theta, &nu);

            if rho < (self.likelihood)(&b)? {
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
