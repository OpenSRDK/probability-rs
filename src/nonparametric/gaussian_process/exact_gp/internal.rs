use super::{super::kernel_matrix::kernel_matrix, ExactGP, GaussianProcessParams};
use crate::DistributionError;
use crate::MultivariateNormalParams;
use crate::RandomVariable;
use opensrdk_kernel_method::*;
pub use rayon::prelude::*;

impl<K, T> ExactGP<K, T>
where
  K: Kernel<T>,
  T: RandomVariable,
{
  pub(crate) fn handle_temporal_params(
    &self,
    params: &GaussianProcessParams<T>,
  ) -> Result<MultivariateNormalParams, DistributionError> {
    let kxx = kernel_matrix(&self.kernel, &params.theta, &params.x, &params.x)?;
    let lkxx = kxx.potrf()?;

    let params = MultivariateNormalParams::new(vec![0.0; params.x.len()], lkxx)?;

    return Ok(params);
  }
}
