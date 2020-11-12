use crate::{Distribution, DistributionParam, DistributionParamVal, LogDistribution};
use opensrdk_linear_algebra::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::{error::Error, f64::consts::PI};

#[derive(Debug)]
pub struct MultivariateNormal {
    mu: Box<dyn DistributionParam<Vec<f64>>>,
    l_sigma: Box<dyn DistributionParam<Matrix>>,
}

#[derive(thiserror::Error, Debug)]
pub enum MultivariateNormalError {
    #[error("dimension mismatch")]
    DimensionMismatch,
}

impl MultivariateNormal {
    /// # Multivariate normal
    /// `L` is needed as second argument under decomposition `Sigma = L * L^T`
    /// l_sigma = sigma.potrf()?;
    pub fn new(
        mu: Box<dyn DistributionParam<Vec<f64>>>,
        l_sigma: Box<dyn DistributionParam<Matrix>>,
    ) -> Result<Self, Box<dyn Error>> {
        let n = mu.value().len();
        if n != l_sigma.value().rows() || n != l_sigma.value().cols() {
            return Err(MultivariateNormalError::DimensionMismatch.into());
        }

        Ok(Self { mu, l_sigma })
    }

    pub fn mu(&self) -> &Vec<f64> {
        self.mu.value()
    }

    pub fn l_sigma(&self) -> &Matrix {
        self.l_sigma.value()
    }
}

impl<'a> Distribution<'a, Vec<f64>> for MultivariateNormal {
    fn p(&self, x: &Vec<f64>) -> Result<f64, Box<dyn Error>> {
        let mu = self.mu();
        let l_sigma = self.l_sigma();

        let n = x.len();

        if n != mu.len() {
            return Err(MultivariateNormalError::DimensionMismatch.into());
        }
        let n = n as f64;

        let x_mu = x
            .par_iter()
            .zip(mu.par_iter())
            .map(|(&xi, &mui)| xi - mui)
            .collect::<Vec<_>>()
            .col_mat();

        Ok(1.0 / ((2.0 * PI).powf(n / 2.0) * l_sigma.trdet())
            * (-1.0 / 2.0 * (x_mu.t() * l_sigma.potrs(x_mu)?)[0][0]).exp())
    }

    fn sample(&self, rng: &mut StdRng) -> Result<Vec<f64>, Box<dyn Error>> {
        let mu = self.mu();
        let l_sigma = self.l_sigma();

        let z = (0..l_sigma.rows())
            .into_iter()
            .map(|_| rng.sample(StandardNormal))
            .collect::<Vec<_>>();

        let y = mu.clone().col_mat().gemm(l_sigma, &z.col_mat(), 1.0, 1.0)?;

        Ok(y.vec())
    }

    fn ln(&'a mut self, x: &'a mut DistributionParamVal<Vec<f64>>) -> LogDistribution<'a> {
        let mut params = vec![];
        if let Some(x) = x.mut_for_optimization() {
            params = x.iter_mut().map(|xi| xi).collect();
        }
        if let Some(mu) = self.mu.mut_for_optimization() {
            params = params
                .into_iter()
                .chain(mu.iter_mut().map(|mui| mui))
                .collect();
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
