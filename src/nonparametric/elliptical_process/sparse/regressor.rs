use super::SparseEllipticalProcessParams;
use crate::nonparametric::EllipticalProcessError;
use crate::DistributionError;
use crate::{
    nonparametric::{kernel_matrix, regressor::GaussianProcessRegressor},
    ExactEllipticalParams, RandomVariable,
};
use opensrdk_kernel_method::Kernel;

impl<K, T> GaussianProcessRegressor<K, T> for SparseEllipticalProcessParams<K, T>
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

        let kuxs = kernel_matrix(&self.base.kernel, &self.base.theta, &self.u, xs)?;
        let kxsu = kuxs.t();
        let kxsxs = kernel_matrix(&self.base.kernel, &self.base.theta, xs, xs)?;
        let qxsxs = &kxsu * self.lkuu.potrs(kuxs.clone())?;
        let kxsu_s_inv_kuxs = &kxsu * self.ls.potrs(kuxs)?;

        let mean = self.mu[0] + &kxsu * &self.s_inv_kux_omega_y;
        let covariance = kxsxs - qxsxs + kxsu_s_inv_kuxs;

        ExactEllipticalParams::new(mean.vec(), covariance.potrf()?)
    }
}
