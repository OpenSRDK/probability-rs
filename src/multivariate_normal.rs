use crate::MultivariateDistribution;
use opensrdk_linear_algebra::Matrix;
use rand::prelude::*;
use rand_distr::StandardNormal;

pub struct MultivariateNormal {
    mean: Vec<f64>,
    l_cov: Matrix,
}

impl MultivariateNormal {
    /// # Multivariate normal
    /// `L` is needed as second argument under decomposition `Sigma = L * L^T`
    pub fn new(mean: Vec<f64>, l_cov: Matrix) -> Self {
        Self { mean, l_cov }
    }

    pub fn from(mean: Vec<f64>, cov: Matrix) -> Result<Self, String> {
        if mean.len() != cov.rows() {
            return Err("dimension mismatch".to_owned());
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
    fn sample(&self, rng: &mut StdRng) -> Result<Vec<f64>, String> {
        let z = (0..self.l_cov.rows())
            .into_iter()
            .map(|_| rng.sample(StandardNormal))
            .collect::<Vec<_>>();

        let y = Matrix::col(self.mean.clone()).gemm(&self.l_cov, &Matrix::col(z), 1.0, 1.0)?;

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
