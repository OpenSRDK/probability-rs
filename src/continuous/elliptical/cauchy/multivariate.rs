use crate::{
    DependentJoint, Distribution, ExactEllipticalParams, IndependentJoint, MultivariateStudentT,
    MultivariateStudentTParams, RandomVariable,
};
use crate::{DistributionError, EllipticalParams};
use rand::prelude::*;
use std::marker::PhantomData;
use std::{ops::BitAnd, ops::Mul};

/// Multivariate cauchy distribution
#[derive(Clone, Debug)]
pub struct MultivariateCauchy<T = ExactEllipticalParams>
where
    T: EllipticalParams,
{
    phantom: PhantomData<T>,
}

impl<T> MultivariateCauchy<T>
where
    T: EllipticalParams,
{
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum MultivariateCauchyError {}

impl<T> Distribution for MultivariateCauchy<T>
where
    T: EllipticalParams,
{
    type Value = Vec<f64>;
    type Condition = T;

    fn fk(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let studentt_params = MultivariateStudentTWrapper::new(theta);

        MultivariateStudentT::new().fk(x, &studentt_params)
    }

    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let studentt_params = MultivariateStudentTWrapper::new(theta);

        MultivariateStudentT::new().sample(&studentt_params, rng)
    }
}

#[derive(Clone, Debug, PartialEq)]
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
    Rhs: Distribution<Value = TRhs, Condition = T>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, Vec<f64>, TRhs, T>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<T, Rhs, URhs> BitAnd<Rhs> for MultivariateCauchy<T>
where
    T: EllipticalParams,
    Rhs: Distribution<Value = T, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, Vec<f64>, T, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Distribution, ExactMultivariateCauchyParams, MultivariateCauchy};
    use opensrdk_linear_algebra::*;
    use rand::prelude::*;
    #[test]
    fn it_works() {
        let cauchy = MultivariateCauchy::new();
        let mut rng = StdRng::from_seed([1; 32]);

        let mu = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let lsigma = mat!(
           1.0,  0.0,  0.0,  0.0,  0.0,  0.0;
           2.0,  3.0,  0.0,  0.0,  0.0,  0.0;
           4.0,  5.0,  6.0,  0.0,  0.0,  0.0;
           7.0,  8.0,  9.0, 10.0,  0.0,  0.0;
          11.0, 12.0, 13.0, 14.0, 15.0,  0.0;
          16.0, 17.0, 18.0, 19.0, 20.0, 21.0
        );
        println!("{:#?}", lsigma);

        let x = cauchy
            .sample(
                &ExactMultivariateCauchyParams::new(mu, lsigma).unwrap(),
                &mut rng,
            )
            .unwrap();

        println!("{:#?}", x);
    }
}
