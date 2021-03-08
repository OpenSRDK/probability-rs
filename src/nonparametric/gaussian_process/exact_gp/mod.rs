use super::{
    ey::ey, ey::y_ey, kernel_matrix::kernel_matrix, GaussianProcessError, GaussianProcessParams,
};
use crate::{opensrdk_linear_algebra::*, Distribution};
use crate::{MultivariateNormal, MultivariateNormalParams};
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
    l_kxx: Matrix,
    kxx_inv_y: Matrix,
}

impl<K, T> ExactGP<K, T>
where
    K: Kernel<T>,
    T: Clone + Debug,
{
    fn reset_prepare(&mut self) -> Result<&mut Self, Box<dyn Error>> {
        self.ready_to_predict = false;
        self.l_kxx = mat!();
        self.kxx_inv_y = mat!();

        Ok(self)
    }

    fn multivariate_normal(
        &self,
        params: &GaussianProcessParams<T>,
    ) -> Result<MultivariateNormalParams, Box<dyn Error>> {
        if params.x.is_none() && params.theta.is_none() {
            let params =
                MultivariateNormalParams::new(vec![self.ey; self.x.len()], self.l_kxx.clone())?;

            return Ok(params);
        }

        let params_x = params.x.as_ref().unwrap_or(&self.x);
        let params_theta = params.theta.as_ref().unwrap_or(&self.theta);
        let kxx = kernel_matrix(&self.kernel, params_theta, params_x, params_x)?;
        let l_kxx = kxx.potrf()?;

        let params = MultivariateNormalParams::new(vec![self.ey; self.x.len()], l_kxx)?;

        return Ok(params);
    }
}

impl<K, T> ExactGP<K, T>
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
            l_kxx: mat!(),
            kxx_inv_y: mat!(),
        }
    }

    fn set_x(&mut self, x: Vec<T>) -> Result<&Self, Box<dyn Error>> {
        self.x = x;
        self.reset_prepare()?;

        Ok(self)
    }

    fn set_theta(&mut self, theta: Vec<f64>) -> Result<&Self, Box<dyn Error>> {
        let params_len = self.kernel.params_len();
        if theta.len() != params_len {
            return Err(GaussianProcessError::DimensionMismatch.into());
        }

        self.theta = theta;
        self.reset_prepare()?;

        Ok(self)
    }

    fn kernel(&self) -> &K {
        &self.kernel
    }

    fn theta(&self) -> &[f64] {
        &self.theta
    }

    fn prepare_predict(&mut self, y: &[f64]) -> Result<(), Box<dyn Error>> {
        let n = self.x.len();
        if n == 0 {
            return Err(GaussianProcessError::Empty.into());
        }

        if n != y.len() {
            return Err(GaussianProcessError::DimensionMismatch.into());
        }

        self.ey = ey(y);
        let y_ey = &y_ey(y, self.ey);

        let kxx = kernel_matrix(&self.kernel, &self.theta, &self.x, &self.x)?;
        self.l_kxx = kxx.potrf()?;
        self.kxx_inv_y = self.l_kxx.potrs(y_ey.to_vec().col_mat())?.vec().col_mat();

        self.ready_to_predict = true;

        Ok(())
    }

    fn predict_multivariate(&self, xs: &[T]) -> Result<MultivariateNormalParams, Box<dyn Error>> {
        if !self.ready_to_predict {
            return Err(GaussianProcessError::NotPrepared.into());
        }

        let kxxs = kernel_matrix(&self.kernel, &self.theta, &self.x, xs)?;
        let kxx_inv_kxxs_t = self.l_kxx.potrs(kxxs.clone())?;
        let kxsxs = kernel_matrix(&self.kernel, &self.theta, xs, xs)?;

        let mean = self.ey + (&self.kxx_inv_y.t() * &kxxs).t();
        let covariance = kxsxs - (kxx_inv_kxxs_t * &kxxs).t();

        MultivariateNormalParams::new(mean.vec(), covariance.potrf()?)
    }
}

impl<K, T> Distribution for ExactGP<K, T>
where
    K: Kernel<T>,
    T: Clone + Debug + PartialEq,
{
    type T = Vec<f64>;
    type U = GaussianProcessParams<T>;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, Box<dyn Error>> {
        let normal = MultivariateNormal;
        let params = self.multivariate_normal(theta)?;

        return Ok(normal.p(x, &params)?);
    }

    fn sample(
        &self,
        theta: &Self::U,
        rng: &mut rand::prelude::StdRng,
    ) -> Result<Self::T, Box<dyn Error>> {
        let normal = MultivariateNormal;
        let params = self.multivariate_normal(theta)?;

        return Ok(normal.sample(&params, rng)?);
    }
}
