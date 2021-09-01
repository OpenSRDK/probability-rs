use super::KissLoveEllipticalProcessParams;
use crate::DistributionError;
use crate::{
    nonparametric::regressor::GaussianProcessRegressor, ExactEllipticalParams, RandomVariable,
};
use opensrdk_kernel_method::{Convolutable, Convolutional, Kernel};

impl<K, T> GaussianProcessRegressor<Convolutional<K>, T> for KissLoveEllipticalProcessParams<K, T>
where
    K: Kernel<Vec<f64>>,
    T: RandomVariable + Convolutable,
{
    fn gp_predict_multivariate(
        &self,
        xs: &[T],
    ) -> Result<ExactEllipticalParams, DistributionError> {
        todo!();
    }
}
