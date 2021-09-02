use super::ExactEllipticalProcessParams;
use crate::{
    nonparametric::{kernel_matrix, regressor::GaussianProcessRegressor, EllipticalProcessError},
    ExactEllipticalParams, RandomVariable,
};
use crate::{DistributionError, EllipticalParams};
use opensrdk_kernel_method::Kernel;

impl<K, T> GaussianProcessRegressor<K, T> for ExactEllipticalProcessParams<K, T>
where
    K: Kernel<T>,
    T: RandomVariable,
{
    fn gp_predict_multivariate(
        &self,
        xs: &[T],
    ) -> Result<ExactEllipticalParams, DistributionError> {
        let len = xs.len();
        if len == 0 {
            return Err(DistributionError::InvalidParameters(
                EllipticalProcessError::Empty.into(),
            ));
        }

        let kxxs = kernel_matrix(&self.base.kernel, &self.base.theta, &self.base.x, xs)?;
        let kxsx = kxxs.t();
        let kxsxs = kernel_matrix(&self.base.kernel, &self.base.theta, xs, xs)?;
        let kxx_inv_kxxs = self.sigma_inv_mul(kxxs)?;

        let mean = self.mu[0] + &kxsx * &self.sigma_inv_y;
        let covariance = kxsxs - &kxsx * kxx_inv_kxxs;

        ExactEllipticalParams::new(mean.vec(), covariance.potrf()?)
    }
}
