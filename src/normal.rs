use crate::{Distribution, DistributionParam, DistributionParamVal, LogDistribution};
use rand::prelude::*;
use rand_distr::Normal as RandNormal;
use std::{error::Error, f64::consts::PI};

#[derive(Debug)]
pub struct Normal {
    mu: Box<dyn DistributionParam<f64>>,
    sigma: Box<dyn DistributionParam<f64>>,
}

#[derive(thiserror::Error, Debug)]
pub enum NormalError {
    #[error("invalid variance")]
    InvalidVariance,
}

impl Normal {
    pub fn new(
        mu: Box<dyn DistributionParam<f64>>,
        sigma: Box<dyn DistributionParam<f64>>,
    ) -> Result<Self, Box<dyn Error>> {
        if *sigma.value() <= 0.0 {
            return Err(NormalError::InvalidVariance.into());
        }

        Ok(Self { mu, sigma })
    }

    pub fn mu(&self) -> &f64 {
        self.mu.value()
    }

    pub fn sigma(&self) -> &f64 {
        self.sigma.value()
    }
}

impl<'a> Distribution<'a, f64> for Normal {
    fn p(&self, x: &f64) -> Result<f64, Box<dyn Error>> {
        let mu = self.mu();
        let sigma = self.sigma();

        Ok(1.0 / (2.0 * PI * sigma.powi(2)).sqrt()
            * (-(x - mu).powi(2) / (2.0 * sigma.powi(2))).exp())
    }

    fn sample(&self, rng: &mut StdRng) -> Result<f64, Box<dyn Error>> {
        let mu = self.mu();
        let sigma = self.sigma();

        let normal = match RandNormal::new(*mu, *sigma) {
            Ok(n) => n,
            Err(_) => return Err(NormalError::InvalidVariance.into()),
        };

        Ok(rng.sample(normal))
    }

    fn ln(&'a mut self, x: &'a mut DistributionParamVal<f64>) -> LogDistribution<'a> {
        let mut params = vec![];
        if let Some(x) = x.mut_for_optimization() {
            params = vec![x];
        }
        if let Some(mu) = self.mu.mut_for_optimization() {
            params.push(mu);
        }
        if let Some(sigma) = self.sigma.mut_for_optimization() {
            params.push(sigma);
        }

        let l = || -> Result<(f64, Vec<(&'a f64, f64)>), Box<dyn Error>> { Ok((0.0, vec![])) };

        LogDistribution::<'a>::new(params, Box::new(l))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
