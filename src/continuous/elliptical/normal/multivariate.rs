use crate::{
    ConditionDifferentiableDistribution, DependentJoint, Distribution, ExactEllipticalParams,
    IndependentJoint, RandomVariable, ValueDifferentiableDistribution,
};
use crate::{DistributionError, EllipticalParams};
use opensrdk_linear_algebra::{DiagonalMatrix, Vector};
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
    type Value = Vec<f64>;
    type Condition = T;

    fn fk(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let x_mu = theta.x_mu(x)?.col_mat();
        let n = x.len();

        // For preventing the result from being zero, dividing e^n
        Ok((-1.0 / 2.0 * (x_mu.t() * theta.sigma_inv_mul(x_mu)?)[(0, 0)] / (n as f64).exp()).exp())
    }

    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let z = (0..theta.lsigma_cols())
            .into_iter()
            .map(|_| rng.sample(StandardNormal))
            .collect::<Vec<f64>>();

        Ok(theta.sample(z)?)
    }
}

impl<T, Rhs, TRhs> Mul<Rhs> for MultivariateNormal<T>
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

impl<T, Rhs, URhs> BitAnd<Rhs> for MultivariateNormal<T>
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

impl ValueDifferentiableDistribution for MultivariateNormal {
    fn ln_diff_value(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let lsigma_mat = theta.lsigma().0.to_mat();
        let sigma = &lsigma_mat * lsigma_mat.t();
        let f = -1.0 * x.clone().row_mat() * sigma;
        Ok(f.vec())
    }
}

impl ConditionDifferentiableDistribution for MultivariateNormal {
    fn ln_diff_condition(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        // let lsigma_mat = theta.lsigma().0.to_mat();
        let lsigma_inv = theta.lsigma().clone().pptri()?.to_mat();
        // let sigma_inv = DiagonalMatrix::new((&lsigma_mat * lsigma_mat.t()).vec())
        //     .powf(-1.0)
        //     .mat();
        let mu_mat = theta.x_mu(x)?.col_mat();
        let x_mat = x.clone().col_mat();
        let x_mu_mat = x_mat - mu_mat;
        let fk = self.fk(x, theta).unwrap();
        let f_mu = &x_mu_mat * &lsigma_inv * &fk;

        let x_mu_t = x_mu_mat.t();
        // todo
        let lsigma_det = 1.0;
        let f_sigma = (&x_mu_mat * &x_mu_t - &lsigma_inv * &lsigma_inv) / (4.0 * lsigma_det);
        Ok(f_mu.vec())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Distribution, ExactMultivariateNormalParams, MultivariateNormal,
        ValueDifferentiableDistribution,
    };
    use opensrdk_linear_algebra::{pp::trf::PPTRF, *};
    use rand::prelude::*;
    #[test]
    fn it_works() {
        let normal = MultivariateNormal::new();
        let mut rng = StdRng::from_seed([1; 32]);

        let mu = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let lsigma = SymmetricPackedMatrix::from_mat(&mat!(
           1.0,  0.0,  0.0,  0.0,  0.0,  0.0;
           2.0,  3.0,  0.0,  0.0,  0.0,  0.0;
           4.0,  5.0,  6.0,  0.0,  0.0,  0.0;
           7.0,  8.0,  9.0, 10.0,  0.0,  0.0;
          11.0, 12.0, 13.0, 14.0, 15.0,  0.0;
          16.0, 17.0, 18.0, 19.0, 20.0, 21.0
        ))
        .unwrap();
        println!("{:#?}", lsigma);

        let x = normal
            .sample(
                &ExactMultivariateNormalParams::new(mu, PPTRF(lsigma)).unwrap(),
                &mut rng,
            )
            .unwrap();

        println!("{:#?}", x);
    }

    #[test]
    fn it_works2() {
        let normal = MultivariateNormal::new();
        let mut _rng = StdRng::from_seed([1; 32]);

        let mu = vec![0.0, 1.0];
        let lsigma = SymmetricPackedMatrix::from_mat(&mat!(
           1.0,  0.0;
           2.0,  1.0
        ))
        .unwrap();

        let x = vec![0.0, 1.0];

        let f = normal.ln_diff_value(
            &x,
            &ExactMultivariateNormalParams::new(mu, PPTRF(lsigma)).unwrap(),
        );
        println!("{:#?}", f);
    }
}
