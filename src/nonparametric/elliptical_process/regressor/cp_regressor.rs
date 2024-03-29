use super::utils::ref_to_slice;
use crate::nonparametric::GaussianProcessRegressor;
use crate::{DistributionError, ExactMultivariateStudentTParams, StudentTParams};
use crate::{MultivariateStudentTParams, RandomVariable};
use opensrdk_kernel_method::PositiveDefiniteKernel;
use opensrdk_linear_algebra::pp::trf::PPTRF;
use opensrdk_linear_algebra::{SymmetricPackedMatrix, Vector};

pub trait CauchyProcessRegressor<K, T>: GaussianProcessRegressor<K, T>
where
    K: PositiveDefiniteKernel<T>,
    T: RandomVariable,
{
    fn cp_predict(&self, xs: &T) -> Result<StudentTParams, DistributionError> {
        let fs = self.cp_predict_multivariate(ref_to_slice(xs))?;

        Ok(StudentTParams::new(
            fs.nu(),
            fs.mu()[0],
            fs.lsigma().0.elems()[0],
        )?)
    }

    fn cp_predict_multivariate(
        &self,
        xs: &[T],
    ) -> Result<ExactMultivariateStudentTParams, DistributionError>;
}

impl<K, T, GPR> CauchyProcessRegressor<K, T> for GPR
where
    K: PositiveDefiniteKernel<T>,
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

        let coefficient = ((1.0 + mahalanobis_squared) / (1 + n) as f64).sqrt();
        let new_lsigma = PPTRF(
            SymmetricPackedMatrix::from(mu.len(), (coefficient * lsigma.0.eject().col_mat()).vec())
                .unwrap(),
        );

        ExactMultivariateStudentTParams::new((1 + n) as f64, mu, new_lsigma)
    }
}
