use crate::MultivariateDistribution;
use opensrdk_linear_algebra::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;

/// covariance matrix must be stored in upper triangle at least
pub struct MultivariateNormal {
    mean: Matrix,
    covariance: Matrix<PositiveSemiDefinite>,
}

impl MultivariateNormal {
    fn new(mean: Matrix, covariance: Matrix<PositiveSemiDefinite>) -> Self {
        Self { mean, covariance }
    }

    pub fn from(mean: Matrix, covariance: Matrix<PositiveSemiDefinite>) -> Result<Self, String> {
        if mean.get_columns() != 1 || mean.get_rows() != covariance.get_rows() {
            Err("dimension mismatch".to_owned())
        } else {
            Ok(Self::new(mean, covariance))
        }
    }

    pub fn get_mean(&self) -> &Matrix {
        &self.mean
    }

    pub fn get_covariance(&self) -> &Matrix {
        &self.covariance
    }
}

impl MultivariateDistribution for MultivariateNormal {
    fn sample(&self, thread_rng: &mut ThreadRng) -> Result<Matrix, String> {
        let mut z = Matrix::<Standard, f64>::zeros(self.mean.get_rows(), 1);
        for i in 0..self.mean.get_rows() {
            z[i][0] = thread_rng.sample(StandardNormal);
        }

        let (u, sigma, _) = self.covariance.svd()?;

        for i in 0..sigma.get_rows() {
            sigma[i][i] = sigma[i][i].sqrt();
        }

        let y = self.mean.clone() + (u * sigma) * z;

        Ok(y)
    }
}
