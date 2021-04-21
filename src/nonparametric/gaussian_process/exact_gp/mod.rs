pub mod distribution;
pub mod internal;
pub mod regressor;

use super::{GaussianProcess, GaussianProcessParams};
use crate::DistributionError;
use crate::{opensrdk_linear_algebra::*, RandomVariable};
use opensrdk_kernel_method::*;
pub use rayon::prelude::*;
use std::{fmt::Debug, marker::PhantomData};

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
  T: RandomVariable,
{
  kernel: K,
  phantom: PhantomData<T>,
}

impl<K, T> GaussianProcess<K, T> for ExactGP<K, T>
where
  K: Kernel<T>,
  T: RandomVariable,
{
  fn new(kernel: K) -> Self {
    Self {
      kernel,
      phantom: PhantomData,
    }
  }

  fn kxx_inv_vec(
    &self,
    vec: Vec<f64>,
    params: &GaussianProcessParams<T>,
    with_det_lkxx: bool,
  ) -> Result<(Vec<f64>, Option<f64>), DistributionError> {
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
  ) -> Result<Vec<f64>, DistributionError> {
    let params = self.handle_temporal_params(params)?;
    let (_, l_sigma) = params.eject();

    Ok((l_sigma * vec.col_mat()).vec())
  }
}
