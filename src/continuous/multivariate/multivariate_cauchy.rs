use crate::{
    DependentJoint, Distribution, ExactEllipticalParams, IndependentJoint, MultivariateStudentT,
    MultivariateStudentTParams, RandomVariable,
};
use crate::{DistributionError, EllipticalParams};
use rand::prelude::*;
use std::{ops::BitAnd, ops::Mul};

/// # MultivariateCauchy
#[derive(Clone, Debug)]
pub struct MultivariateCauchy<T = ExactEllipticalParams>
where
    T: EllipticalParams;

#[derive(thiserror::Error, Debug)]
pub enum MultivariateCauchyError {}

impl<T> Distribution for MultivariateCauchy<T>
where
    T: EllipticalParams,
{
    type T = Vec<f64>;
    type U = T;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let studentt_params = MultivariateStudentTWrapper::new(theta);

        MultivariateStudentT::p(self, x, studentt_params)
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, DistributionError> {
        let studentt_params = MultivariateStudentTWrapper::new(theta);

        MultivariateStudentT::p(self, studentt_params, rng)
    }
}

struct MultivariateStudentTWrapper<'a, T>
where
    T: EllipticalParams,
{
    elliptical: &'a T,
}

impl<'a, T> MultivariateStudentTWrapper<'a, T>
where
    T: EllipticalParams,
{
    fn new(elliptical: &'a T) -> Self {
        Self { elliptical }
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

impl<T, Rhs, TRhs> Mul<Rhs> for MultivariateCauchy<T>
where
    T: EllipticalParams,
    Rhs: Distribution<T = TRhs, U = T>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, Vec<f64>, TRhs, T>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<T, Rhs, URhs> BitAnd<Rhs> for MultivariateCauchy
where
    T: EllipticalParams,
    Rhs: Distribution<T = T, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, Vec<f64>, T, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
