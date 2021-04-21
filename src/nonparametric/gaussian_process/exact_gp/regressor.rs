use super::super::{ey::ey, ey::y_ey};
use super::ExactGP;
use crate::DistributionError;
use crate::{
  nonparametric::{kernel_matrix, regressor::GaussianProcessRegressor},
  RandomVariable,
};
use crate::{
  nonparametric::{GaussianProcessParams, GaussianProcessRegressorError},
  MultivariateNormalParams,
};
use opensrdk_kernel_method::Kernel;
use opensrdk_linear_algebra::*;

#[derive(Clone, Debug)]
pub struct ExactGPRegressor<K, T>
where
  K: Kernel<T>,
  T: RandomVariable,
{
  gp: ExactGP<K, T>,
  ey: f64,
  x: Vec<T>,
  theta: Vec<f64>,
  lkxx: Matrix,
  kxx_inv_y: Matrix,
}

impl<K, T> GaussianProcessRegressor<ExactGP<K, T>, K, T> for ExactGPRegressor<K, T>
where
  K: Kernel<T>,
  T: RandomVariable,
{
  fn new(
    gp: ExactGP<K, T>,
    y: &[f64],
    params: GaussianProcessParams<T>,
  ) -> Result<Self, DistributionError> {
    let (x, theta) = params.eject();

    let n = y.len();
    if n == 0 {
      return Err(DistributionError::InvalidParameters(
        GaussianProcessRegressorError::Empty.into(),
      ));
    }

    if n != x.len() {
      return Err(DistributionError::InvalidParameters(
        GaussianProcessRegressorError::DimensionMismatch.into(),
      ));
    }

    let ey = ey(y);
    let y_ey = &y_ey(y, ey);

    let kxx = kernel_matrix(&gp.kernel, &theta, &x, &x)?;
    let lkxx = kxx.potrf()?;
    let kxx_inv_y = lkxx.potrs(y_ey.to_vec().col_mat())?.vec().col_mat();

    Ok(Self {
      gp,
      ey,
      x,
      theta,
      lkxx,
      kxx_inv_y,
    })
  }

  fn n(&self) -> usize {
    self.x.len()
  }

  fn ey(&self) -> f64 {
    self.ey
  }

  fn predict_multivariate(&self, xs: &[T]) -> Result<MultivariateNormalParams, DistributionError> {
    let kxxs = kernel_matrix(&self.gp.kernel, &self.theta, &self.x, xs)?;
    let kxx_inv_kxxs_t = self.lkxx.potrs(kxxs.clone())?;
    let kxsxs = kernel_matrix(&self.gp.kernel, &self.theta, xs, xs)?;

    let mean = self.ey + (&self.kxx_inv_y.t() * &kxxs).t();
    let covariance = kxsxs - (kxx_inv_kxxs_t * &kxxs).t();

    MultivariateNormalParams::new(mean.vec(), covariance.potrf()?)
  }
}
