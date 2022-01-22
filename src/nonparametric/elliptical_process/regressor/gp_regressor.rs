use super::utils::ref_to_slice;
use crate::nonparametric::EllipticalProcessParams;
use crate::{DistributionError, NormalParams};
use crate::{ExactEllipticalParams, RandomVariable};
use opensrdk_kernel_method::PositiveDefiniteKernel;

pub trait GaussianProcessRegressor<K, T>: EllipticalProcessParams<K, T>
where
    K: PositiveDefiniteKernel<T>,
    T: RandomVariable,
{
    fn gp_predict(&self, xs: &T) -> Result<NormalParams, DistributionError> {
        let fs = self.gp_predict_multivariate(ref_to_slice(xs))?;

        NormalParams::new(fs.mu()[0], fs.lsigma().0.elems()[0])
    }

    fn gp_predict_multivariate(&self, xs: &[T])
        -> Result<ExactEllipticalParams, DistributionError>;
}
