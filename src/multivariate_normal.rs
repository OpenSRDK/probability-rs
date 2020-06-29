use crate::distribution::MultivariateDistribution;
use opensrdk_linear_algebra::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;

/// covariance matrix must be stored in upper triangle at least
pub struct MultivariateNormal {
    pub mean: Matrix,
    pub covariance: Matrix<PositiveSemiDefinite>,
}

impl MultivariateNormal {
    pub fn new(mean: Matrix, covariance: Matrix<PositiveSemiDefinite>) -> Self {
        Self { mean, covariance }
    }

    pub fn from(mean: Matrix, covariance: Matrix<PositiveSemiDefinite>) -> Result<Self, ()> {
        if mean.get_columns() != 1 || mean.get_rows() != covariance.get_rows() {
            Err(())
        } else {
            Ok(Self::new(mean, covariance))
        }
    }
}

impl MultivariateDistribution for MultivariateNormal {
    fn sample(&self, thread_rng: &mut ThreadRng) -> Matrix {
        let mut z = Matrix::<Standard, f64>::zeros(self.mean.get_rows(), 1);
        for i in 0..self.mean.get_rows() {
            z[i][0] = thread_rng.sample(StandardNormal);
        }

        let (u, sigma, _) = self.covariance.singular_value_decomposition().unwrap();

        for i in 0..sigma.get_rows() {
            sigma[i][i] = sigma[i][i].sqrt();
        }

        let y = self.mean.clone() + (u * sigma) * z;

        y
    }
}
