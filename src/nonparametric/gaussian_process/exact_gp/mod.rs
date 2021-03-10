pub mod distribution;
pub mod internal;

use super::{
    ey::ey, ey::y_ey, kernel_matrix::kernel_matrix, GaussianProcess, GaussianProcessError,
    GaussianProcessParams,
};
use crate::opensrdk_linear_algebra::*;
use crate::MultivariateNormalParams;
use opensrdk_kernel_method::*;
pub use rayon::prelude::*;
use std::{error::Error, fmt::Debug};

/// Gaussian Process without approximation for scalability.
///
///
/// |                 | order                                                   |
/// | --------------- | ------------------------------------------------------- |
/// | pre-computation | ![tex](https://latex.codecogs.com/svg.latex?O%28N^3%29) |
/// | prediction      | ![tex](https://latex.codecogs.com/svg.latex?O%28N^2%29) |
///
/// | type args | mathematical expression                                 |
/// | --------- | ------------------------------------------------------- |
/// | `T`       | ![tex](https://latex.codecogs.com/svg.latex?\mathbb{D}) |
///
#[derive(Clone, Debug)]
pub struct ExactGP<K, T>
where
    K: Kernel<T>,
    T: Clone + Debug,
{
    kernel: K,
    theta: Vec<f64>,
    ready_to_predict: bool,
    ey: f64,
    x: Vec<T>,
    lkxx: Matrix,
    kxx_inv_y: Matrix,
}

impl<K, T> GaussianProcess<K, T> for ExactGP<K, T>
where
    K: Kernel<T>,
    T: Clone + Debug,
{
    fn new(kernel: K) -> Self {
        let params_len = kernel.params_len();
        Self {
            kernel,
            theta: vec![1.0; params_len],
            ready_to_predict: false,
            ey: 0.0,
            x: vec![],
            lkxx: mat!(),
            kxx_inv_y: mat!(),
        }
    }

    fn set_x(&mut self, x: Vec<T>) -> Result<&mut Self, Box<dyn Error>> {
        self.x = x;
        self.reset_prepare();

        Ok(self)
    }

    fn set_theta(&mut self, theta: Vec<f64>) -> Result<&mut Self, Box<dyn Error>> {
        let params_len = self.kernel.params_len();
        if theta.len() != params_len {
            return Err(GaussianProcessError::DimensionMismatch.into());
        }

        self.theta = theta;
        self.reset_prepare();

        Ok(self)
    }

    fn kernel(&self) -> &K {
        &self.kernel
    }

    fn theta(&self) -> &[f64] {
        &self.theta
    }

    fn n(&self) -> usize {
        self.x.len()
    }

    fn ey(&self) -> f64 {
        self.ey
    }

    fn prepare_predict(&mut self, y: &[f64]) -> Result<(), Box<dyn Error>> {
        let n = self.n();
        if n == 0 {
            return Err(GaussianProcessError::Empty.into());
        }

        if n != y.len() {
            return Err(GaussianProcessError::DimensionMismatch.into());
        }

        self.ey = ey(y);
        let y_ey = &y_ey(y, self.ey);

        let kxx = kernel_matrix(&self.kernel, &self.theta, &self.x, &self.x)?;
        self.lkxx = kxx.potrf()?;
        self.kxx_inv_y = self.lkxx.potrs(y_ey.to_vec().col_mat())?.vec().col_mat();

        self.ready_to_predict = true;

        Ok(())
    }

    fn predict_multivariate(&self, xs: &[T]) -> Result<MultivariateNormalParams, Box<dyn Error>> {
        if !self.ready_to_predict {
            return Err(GaussianProcessError::NotPrepared.into());
        }

        let kxxs = kernel_matrix(&self.kernel, &self.theta, &self.x, xs)?;
        let kxx_inv_kxxs_t = self.lkxx.potrs(kxxs.clone())?;
        let kxsxs = kernel_matrix(&self.kernel, &self.theta, xs, xs)?;

        let mean = self.ey + (&self.kxx_inv_y.t() * &kxxs).t();
        let covariance = kxsxs - (kxx_inv_kxxs_t * &kxxs).t();

        MultivariateNormalParams::new(mean.vec(), covariance.potrf()?)
    }

    fn kxx_inv_vec(
        &self,
        vec: Vec<f64>,
        params: &GaussianProcessParams<T>,
        with_det_lkxx: bool,
    ) -> Result<(Vec<f64>, Option<f64>), Box<dyn Error>> {
        let params = self.handle_temporal_params(params)?;
        let (_, lsigma) = params.eject();

        let det = if with_det_lkxx {
            Some(lsigma.trdet())
        } else {
            None
        };
        let kxx_inv_vec = lsigma.potrs(vec.col_mat())?.vec();

        Ok((kxx_inv_vec, det))
    }

    fn lkxx_vec(
        &self,
        vec: Vec<f64>,
        params: &GaussianProcessParams<T>,
    ) -> Result<Vec<f64>, Box<dyn Error>> {
        let params = self.handle_temporal_params(params)?;
        let (_, l_sigma) = params.eject();

        Ok((l_sigma * vec.col_mat()).vec())
    }
}
