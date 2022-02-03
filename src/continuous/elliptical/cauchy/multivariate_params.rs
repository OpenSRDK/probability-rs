use crate::{DistributionError, EllipticalParams};
use crate::{ExactEllipticalParams, MultivariateStudentTParams, RandomVariable};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MultivariateStudentTWrapper<'a, T>
where
    T: EllipticalParams,
{
    elliptical: &'a T,
}

impl<'a, T> MultivariateStudentTWrapper<'a, T>
where
    T: EllipticalParams,
{
    pub(crate) fn new(elliptical: &'a T) -> Self {
        Self { elliptical }
    }
}

impl<'a, T> RandomVariable for MultivariateStudentTWrapper<'a, T>
where
    T: EllipticalParams,
{
    type RestoreInfo = T::RestoreInfo;

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        let t = self.transform_vec();
        (t.0, t.1)
    }

    fn restore(v: &[f64], info: Self::RestoreInfo) -> Result<Self, DistributionError> {
        Ok(MultivariateStudentTWrapper::restore(v, info)?)
    }
}

impl<'a, T> MultivariateStudentTParams<T> for MultivariateStudentTWrapper<'a, T>
where
    T: EllipticalParams,
{
    fn nu(&self) -> f64 {
        1.0
    }

    fn elliptical(&self) -> &T {
        self.elliptical
    }
}

pub type ExactMultivariateCauchyParams = ExactEllipticalParams;
