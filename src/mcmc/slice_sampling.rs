use crate::{Distribution, DistributionError, RandomVariable};
use rand::prelude::*;

/// Sample b from posterior p(b|a) with likelihood p(a|b) and prior p(b)
/// Unbounded Slice Sampling
/// http://chasen.org/~daiti-m/diary/?201510
pub struct SliceSampler<L, P, A>
where
    L: Distribution<T = A, U = f64>,
    P: Distribution<T = f64, U = ()>,
    A: RandomVariable,
{
    value: A,
    likelihood: L,
    prior: P,
}

#[derive(thiserror::Error, Debug)]
pub enum SliceSamplingError {
    #[error("out of range")]
    OutOfRange,
    #[error("Unknown error")]
    Unknown,
}

impl<L, P, A> SliceSampler<L, P, A>
where
    L: Distribution<T = A, U = f64>,
    P: Distribution<T = f64, U = ()>,
    A: RandomVariable,
{
    pub fn new(value: A, likelihood: L, prior: P) -> Result<Self, DistributionError> {
        Ok(Self {
            value,
            likelihood,
            prior,
        })
    }

    pub fn sample(
        &self,
        x: f64,
        max_iter: usize,
        rng: &mut dyn RngCore,
    ) -> Result<f64, DistributionError> {
        let mut st = 0.0;
        let mut ed = 1.0;

        let r = shrink(x)?;
        let slice = self.likelihood.fk(&self.value, &x)? * self.prior.fk(&x, &())?
            - 2.0 * rng.gen_range(0.0f64..1.0f64).ln();

        for _iter in 0..max_iter {
            let rnew = rng.gen_range(st..ed);
            let expanded = expand(rnew)?;

            let newlik = self.likelihood.fk(&self.value, &expanded)?
                * self.prior.fk(&expanded, &())?
                - (2.0 * rnew * (1.0 - rnew));

            if newlik > slice {
                return expand(rnew);
            } else if rnew > r {
                ed = rnew;
            } else if rnew < r {
                st = rnew;
            } else {
                return Ok(x);
            }
        }
        Ok(x)
    }
}

fn expand(p: f64) -> Result<f64, DistributionError> {
    Ok(-100.0 * (1.0 / (p - 1.0)).ln())
}

fn shrink(x: f64) -> Result<f64, DistributionError> {
    Ok(1.0 / (1.0 + (-x / 100.0).exp()))
}
