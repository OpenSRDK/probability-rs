use super::KissLoveEllipticalProcessParams;
use crate::nonparametric::EllipticalProcessError;
use crate::{
    nonparametric::regressor::GaussianProcessRegressor, ExactEllipticalParams, RandomVariable,
};
use crate::{DistributionError, ExactMultivariateNormalParams};
use opensrdk_kernel_method::{Convolutable, Convolutional, Kernel};
use opensrdk_linear_algebra::*;

impl<K, T> GaussianProcessRegressor<Convolutional<K>, T> for KissLoveEllipticalProcessParams<K, T>
where
    K: Kernel<Vec<f64>>,
    T: RandomVariable + Convolutable,
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

        let wxs = &self.u.interpolation_weight(xs)?;
        let p = self.a.len();

        if p != wxs.len() {
            return Err(DistributionError::InvalidParameters(
                EllipticalProcessError::DimensionMismatch.into(),
            ));
        }

        let (mu, lsigma) = (0..p)
            .into_iter()
            .map(|pi| {
                let wxspi = &wxs[pi];
                let wxspit = &wxspi.t();

                let api = &self.a[pi];
                let spi = &self.s[pi];

                let mupi = (wxspit * api).vec();
                let l_sigma_pi = wxspit * spi;

                Ok((mupi, l_sigma_pi))
            })
            .try_fold::<(Vec<f64>, Matrix), _, Result<(Vec<f64>, Matrix), DistributionError>>(
                (self.mu.clone(), Matrix::new(len, len)),
                |a, b: Result<(Vec<f64>, Matrix), DistributionError>| {
                    let b = b?;
                    Ok(((a.0.col_mat() + b.0.col_mat()).vec(), a.1 + b.1))
                },
            )?;

        ExactMultivariateNormalParams::new(mu, lsigma)
    }
}
