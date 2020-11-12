use crate::Distribution;
use rand::prelude::*;
use rayon::prelude::*;
use std::{error::Error, f64::consts::PI, fmt::Debug, marker::PhantomData};

pub trait EllipticalSliceable: Clone + Debug {
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
pub struct EllipticalSliceSampler<'a, T, DL, DP>
where
    T: EllipticalSliceable,
    DL: Distribution<'a, T>,
    DP: Distribution<'a, T>,
{
    likelihood: &'a DL,
    prior: &'a DP,
    phantom: PhantomData<T>,
}

impl<'a, T, DL, DP> EllipticalSliceSampler<'a, T, DL, DP>
where
    T: EllipticalSliceable,
    DL: Distribution<'a, T>,
    DP: Distribution<'a, T>,
{
    pub fn new(likelihood: &'a DL, prior: &'a DP) -> Self {
        Self {
            likelihood,
            prior,
            phantom: PhantomData,
        }
    }

    pub fn sample(&self, rng: &mut StdRng) -> Result<T, Box<dyn Error>> {
        let nu = self.prior.sample(rng)?;

        let mut b = self.prior.sample(rng)?;
        let rho = self.likelihood.p(&b)? * rng.gen_range(0.0, 1.0);
        let mut theta = rng.gen_range(0.0, 2.0 * PI);

        let mut start = theta - 2.0 * PI;
        let mut end = theta;

        loop {
            b = b.ellipse(theta, &nu);

            if rho < self.likelihood.p(&b)? {
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
