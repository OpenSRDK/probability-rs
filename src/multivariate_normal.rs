use crate::Distribution;
use opensrdk_linear_algebra::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::{error::Error, f64::consts::PI};

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
            return Err(MultivariateNormalError::DimensionMismatch.into());
        }

        let l_cov = cov.potrf()?;

        Ok(Self::new(mean, l_cov))
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

impl Distribution<Vec<f64>> for MultivariateNormal {
    fn p(&self, x: Vec<f64>) -> Result<f64, Box<dyn Error>> {
        let n = x.len() as f64;
        let x_mu_t = x.row_mat() - self.mean.clone().row_mat();

        Ok(1.0 / ((2.0 * PI).powf(n / 2.0) * self.l_cov.trdet())
            * (-1.0 / 2.0 * (&x_mu_t * self.l_cov.potrs(x_mu_t.clone())?.elems().col_mat())[0][0])
                .exp())
    }

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
