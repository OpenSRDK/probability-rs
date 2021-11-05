use crate::{
    DependentJoint, Distribution, ExactEllipticalParams, IndependentJoint, RandomVariable,
};
use crate::{DistributionError, EllipticalParams};
use opensrdk_linear_algebra::Vector;
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::marker::PhantomData;
use std::{ops::BitAnd, ops::Mul};

/// Multivariate normal distribution
#[derive(Clone, Debug)]
pub struct MultivariateNormal<T = ExactEllipticalParams>
where
    T: EllipticalParams,
{
    phantom: PhantomData<T>,
}

impl<T> MultivariateNormal<T>
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
pub enum MultivariateNormalError {}

impl<T> Distribution for MultivariateNormal<T>
where
    T: EllipticalParams,
{
    type T = Vec<f64>;
    type U = T;

    fn fk(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let x_mu = theta.x_mu(x)?.col_mat();
        let n = x.len();

        // For preventing the result from being zero, dividing e^n
        Ok((-1.0 / 2.0 * (x_mu.t() * theta.sigma_inv_mul(x_mu)?)[(0, 0)] / (n as f64).exp()).exp())
    }

    fn sample(&self, theta: &Self::U, rng: &mut dyn RngCore) -> Result<Self::T, DistributionError> {
        let z = (0..theta.lsigma_cols())
            .into_iter()
            .map(|_| rng.sample(StandardNormal))
            .collect::<Vec<f64>>();

        Ok(theta.sample(z)?)
    }
}

pub type ExactMultivariateNormalParams = ExactEllipticalParams;

impl<T, Rhs, TRhs> Mul<Rhs> for MultivariateNormal<T>
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

impl<T, Rhs, URhs> BitAnd<Rhs> for MultivariateNormal<T>
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
    use crate::{Distribution, ExactMultivariateNormalParams, MultivariateNormal};
    use opensrdk_linear_algebra::*;
    use rand::prelude::*;
    #[test]
    fn it_works() {
        let normal = MultivariateNormal::new();
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

        let x = normal
            .sample(
                &ExactMultivariateNormalParams::new(mu, lsigma).unwrap(),
                &mut rng,
            )
            .unwrap();

        println!("{:#?}", x);
    }
}
