use crate::MultivariateDistribution;
use opensrdk_linear_algebra::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::error::Error;

pub struct MultivariateNormal {
    mean: Vec<f64>,
    l_cov: Matrix,
}

#[derive(thiserror::Error, Debug)]
pub enum MultivariateNormalError {
    #[error("dimension mismatch")]
    DimensionMismatch,
}

impl MultivariateNormal {
    /// # Multivariate normal
    /// `L` is needed as second argument under decomposition `Sigma = L * L^T`
    pub fn new(mean: Vec<f64>, l_cov: Matrix) -> Self {
        Self { mean, l_cov }
    }

    pub fn from(mean: Vec<f64>, cov: Matrix) -> Result<Self, Box<dyn Error>> {
        if mean.len() != cov.rows() {
            return Err(Box::new(MultivariateNormalError::DimensionMismatch));
        }

        let decomposed = {
            let (u, sigma, _) = cov.gesvd()?;

            u * sigma.dipowf(0.5)
        };

        Ok(Self::new(mean, decomposed))
    }

    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    pub fn l_cov(&self) -> &Matrix {
        &self.l_cov
    }

    pub fn cov(&self) -> Matrix {
        &self.l_cov * self.l_cov.t()
    }
}

impl MultivariateDistribution for MultivariateNormal {
    fn sample(&self, rng: &mut StdRng) -> Result<Vec<f64>, Box<dyn Error>> {
        let z = (0..self.l_cov.rows())
            .into_iter()
            .map(|_| rng.sample(StandardNormal))
            .collect::<Vec<_>>();

        let y = self
            .mean
            .clone()
            .col_mat()
            .gemm(&self.l_cov, &z.col_mat(), 1.0, 1.0)?;

        Ok(y.elems())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
