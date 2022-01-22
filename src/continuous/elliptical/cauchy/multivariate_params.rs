use crate::EllipticalParams;
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
    type RestoreInfo = usize;

    fn transform_vec(self) -> (Vec<f64>, Self::RestoreInfo) {
        todo!()
    }

    fn restore(v: Vec<f64>, info: Self::RestoreInfo) -> Self {
        todo!()
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
