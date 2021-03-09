use super::{super::kernel_matrix::kernel_matrix, ExactGP, GaussianProcessParams};
use crate::opensrdk_linear_algebra::*;
use crate::MultivariateNormalParams;
use opensrdk_kernel_method::*;
pub use rayon::prelude::*;
use std::{error::Error, fmt::Debug};

impl<K, T> ExactGP<K, T>
where
    K: Kernel<T>,
    T: Clone + Debug,
{
    pub(crate) fn reset_prepare(&mut self) {
        self.ready_to_predict = false;
        self.l_kxx = mat!();
        self.kxx_inv_y = mat!();
    }

    pub(crate) fn handle_temporal_params(
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
