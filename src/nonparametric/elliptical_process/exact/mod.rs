pub mod regressor;

pub use regressor::*;

use super::{BaseEllipticalProcessParams, EllipticalProcessParams};
use crate::nonparametric::{ey, kernel_matrix, EllipticalProcessError};
use crate::{opensrdk_linear_algebra::*, RandomVariable};
use crate::{DistributionError, EllipticalParams};
use ey::y_ey;
use opensrdk_kernel_method::*;
use opensrdk_linear_algebra::matrix::ge::sy_he::po::trf::POTRF;

#[derive(Clone, Debug)]
pub struct ExactEllipticalProcessParams<K, T>
where
    K: PositiveDefiniteKernel<T>,
    T: RandomVariable,
{
    pub base: BaseEllipticalProcessParams<K, T>,
    pub mu: Vec<f64>,
    pub lsigma: POTRF,
    pub sigma_inv_y: Matrix,
    pub mahalanobis_squared: f64,
}

impl<K, T> ExactEllipticalProcessParams<K, T>
where
    K: PositiveDefiniteKernel<T>,
    T: RandomVariable,
{
    fn new(base: BaseEllipticalProcessParams<K, T>, y: &[f64]) -> Result<Self, DistributionError> {
        let n = y.len();
        if n == 0 {
            return Err(DistributionError::InvalidParameters(
                EllipticalProcessError::Empty.into(),
            ));
        }
        if n != base.x.len() {
            return Err(DistributionError::InvalidParameters(
                EllipticalProcessError::DimensionMismatch.into(),
            ));
        }

        let ey = ey(y);
        let mu = vec![ey; base.x.len()];
        let kxx = kernel_matrix(&base.kernel, &base.theta, &base.x, &base.x)?;
        let sigma = kxx + vec![base.sigma.powi(2); n].diag();
        let lsigma = sigma.potrf()?;
        let y_ey = y_ey(y, ey).col_mat();
        let y_ey_t = y_ey.t();
        let sigma_inv_y = lsigma.potrs(y_ey)?;
        let mahalanobis_squared = (y_ey_t * &sigma_inv_y)[(0, 0)];

        Ok(Self {
            base,
            mu,
            lsigma,
            sigma_inv_y,
            mahalanobis_squared,
        })
    }
}

impl<K, T> BaseEllipticalProcessParams<K, T>
where
    K: PositiveDefiniteKernel<T>,
    T: RandomVariable,
{
    /// Elliptical Process without approximation for scalability.
    ///
    /// - Pre-computation time: O(n^3)
    /// - Pre-computation storage: O(n^2)
    /// - Prediction time: O(n^2)
    pub fn exact(self, y: &[f64]) -> Result<ExactEllipticalProcessParams<K, T>, DistributionError> {
        ExactEllipticalProcessParams::new(self, y)
    }
}

impl<K, T> RandomVariable for ExactEllipticalProcessParams<K, T>
where
    K: PositiveDefiniteKernel<T>,
    T: RandomVariable,
{
    type RestoreInfo = ();

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        todo!()
    }

    fn len(&self) -> usize {
        todo!()
    }

    fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        todo!()
    }
}

impl<K, T> EllipticalParams for ExactEllipticalProcessParams<K, T>
where
    K: PositiveDefiniteKernel<T>,
    T: RandomVariable,
{
    fn mu(&self) -> &Vec<f64> {
        &self.mu
    }

    fn sigma_inv_mul(&self, v: Matrix) -> Result<Matrix, DistributionError> {
        Ok(self.lsigma.potrs(v)?)
    }

    fn lsigma_cols(&self) -> usize {
        self.lsigma.0.cols()
    }

    fn sample(&self, z: Vec<f64>) -> Result<Vec<f64>, DistributionError> {
        Ok((self.mu[0] + &self.lsigma.0 * z.col_mat()).vec())
    }
}

impl<K, T> EllipticalProcessParams<K, T> for ExactEllipticalProcessParams<K, T>
where
    K: PositiveDefiniteKernel<T>,
    T: RandomVariable,
{
    fn mahalanobis_squared(&self) -> f64 {
        self.mahalanobis_squared
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        nonparametric::BaseEllipticalProcessParams, ConditionDifferentiableDistribution,
        Distribution, ExactMultivariateNormalParams, MultivariateNormal,
        ValueDifferentiableDistribution,
    };
    use opensrdk_kernel_method::*;
    use opensrdk_linear_algebra::{pp::trf::PPTRF, *};
    use rand::prelude::*;
    use rand_distr::StandardNormal;
    use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

    #[test]
    fn it_works() {
        let normal = MultivariateNormal::new();
        let mut _rng = StdRng::from_seed([1; 32]);

        let samples = samples(7);
        let x = samples.par_iter().map(|v| vec![v.0]).collect::<Vec<_>>();
        let y = samples.par_iter().map(|v| v.1).collect::<Vec<_>>();
        let y2 = vec![1.0; y.len()];
        let kernel = RBF;
        let theta = vec![1.0; kernel.params_len()];
        let sigma = 2.0;

        let params = &BaseEllipticalProcessParams::new(kernel, x, theta, sigma)
            .unwrap()
            .exact(&y)
            .unwrap();

        let f = normal.ln_diff_condition(&y2, params).unwrap();
        println!("{:#?}", f);
    }

    fn func(x: f64) -> f64 {
        0.1 * x + x.sin() + 2.0 * (-x.powi(2)).exp()
    }

    fn samples(size: usize) -> Vec<(f64, f64)> {
        let mut rng = StdRng::from_seed([1; 32]);
        let mut rng2 = StdRng::from_seed([32; 32]);

        (0..size)
            .into_iter()
            .map(|_| {
                let x = rng2.gen_range(-8.0..=8.0);
                let y = func(x) + rng.sample::<f64, _>(StandardNormal);

                (x, y)
            })
            .collect()
    }
}
