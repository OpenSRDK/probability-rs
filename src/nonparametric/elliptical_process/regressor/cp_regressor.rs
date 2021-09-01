use super::utils::ref_to_slice;
use crate::nonparametric::{EllipticalProcessParams, GaussianProcessRegressor};
use crate::RandomVariable;
use crate::{DistributionError, ExactMultivariateStudentTParams, StudentTParams};
use opensrdk_kernel_method::Kernel;

pub trait CauchyProcessRegressor<K, T>: EllipticalProcessParams<K, T>
where
    K: Kernel<T>,
    T: RandomVariable,
{
    fn cp_predict(&self, xs: &T) -> Result<StudentTParams, DistributionError> {
        let fs = self.predict_multivariate(ref_to_slice(xs))?;

        Ok(StudentTParams::new(fs.nu(), fs.mu()[0], fs.lsigma()[0][0]))
    }

    fn cp_predict_multivariate(
        &self,
        xs: &[T],
    ) -> Result<ExactMultivariateStudentTParams, DistributionError>;
}

impl<K, T, GPR> CauchyProcessRegressor<K, T> for GPR
where
    K: Kernel<T>,
    T: RandomVariable,
    GPR: GaussianProcessRegressor<K, T>,
{
    fn cp_predict_multivariate(
        &self,
        xs: &[T],
    ) -> Result<ExactMultivariateStudentTParams, DistributionError> {
        let n = self.mu().len();
        let mahalanobis_squared = self.mahalanobis_squared();

        let (mu, lsigma) = self.gp_predict_multivariate(xs)?.eject();

        Ok(ExactMultivariateStudentTParams::new(
            1 + n,
            mu,
            ((1.0 + mahalanobis_squared) / (1 + n) as f64).sqrt() * lsigma,
        ))
    }
}
