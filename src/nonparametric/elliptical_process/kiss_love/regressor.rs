use super::KissLoveEllipticalProcessParams;
use crate::DistributionError;
use crate::{
    nonparametric::{kernel_matrix, regressor::GaussianProcessRegressor},
    ExactEllipticalParams, RandomVariable,
};
use opensrdk_kernel_method::Kernel;

impl<K, T> GaussianProcessRegressor<K, T> for KissLoveEllipticalProcessParams<K, T>
where
    K: Kernel<T>,
    T: RandomVariable,
{
    fn gp_predict_multivariate(
        &self,
        xs: &[T],
    ) -> Result<ExactEllipticalParams, DistributionError> {
        todo!();
    }
}
