use crate::nonparametric::ExactEllipticalProcessParams;
use crate::{
    ConditionDifferentiableDistribution, DependentJoint, Distribution, ExactEllipticalParams,
    IndependentJoint, RandomVariable, SampleableDistribution, ValueDifferentiableDistribution,
};
use crate::{DistributionError, EllipticalParams};
use opensrdk_kernel_method::PositiveDefiniteKernel;
use opensrdk_linear_algebra::{DiagonalMatrix, Matrix, SymmetricPackedMatrix, Vector};
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

    fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let x_mu = theta.x_mu(x)?.col_mat();
        let n = x.len();

        // For preventing the result from being zero, dividing e^n
        Ok((-1.0 / 2.0 * (x_mu.t() * theta.sigma_inv_mul(x_mu)?)[(0, 0)] / (n as f64).exp()).exp())
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

impl<T> SampleableDistribution for MultivariateNormal<T>
where
    T: EllipticalParams,
{
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

impl<T> ValueDifferentiableDistribution for MultivariateNormal<T>
where
    T: EllipticalParams,
{
    fn ln_diff_value(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let mu_mat = theta.x_mu(x)?.row_mat();
        let x_mat = x.clone().row_mat();
        let x_mu_mat = x_mat - mu_mat;
        let f_x = theta.sigma_inv_mul(x_mu_mat).unwrap();
        Ok(f_x.vec())
    }
}

impl ConditionDifferentiableDistribution for MultivariateNormal {
    fn ln_diff_condition(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let x_mu_mat = theta.x_mu(x)?.col_mat();
        // let x_mat = x.clone().col_mat();
        // let x_mu_mat = x_mat - mu_mat;
        let n = theta.lsigma_cols();
        let identity = DiagonalMatrix::<f64>::identity(n).mat();
        let sigma_inv = theta.sigma_inv_mul(identity).unwrap();
        let sigma_inv_t = sigma_inv.t();
        let f_mu = 0.5 * ((sigma_inv + sigma_inv_t.clone()) * x_mu_mat.clone());

        let x_mu_t = x_mu_mat.t();
        println!("x_mu_t{:?}", x_mu_t);
        let sigma_inv_mul_x_mu = theta.sigma_inv_mul(x_mu_t.clone()).unwrap();
        let sigma_inv_mul_sigma_inv_mul_x_mu = theta.sigma_inv_mul(sigma_inv_mul_x_mu).unwrap();
        let x_mul_sigma = x_mu_mat * sigma_inv_mul_sigma_inv_mul_x_mu;

        let lsigma = theta.lsigma.0.to_mat();

        let f_sigma = (x_mul_sigma - sigma_inv_t) * lsigma;
        println!("f_sigma{:?}", f_sigma);
        let f_mu_vec = f_mu.vec();
        let f_sigma_vec = f_sigma.vec();
        let result_orig = [f_mu_vec, f_sigma_vec];
        let result = result_orig.concat();

        Ok(result)
    }
}

impl<K, T> ConditionDifferentiableDistribution
    for MultivariateNormal<ExactEllipticalProcessParams<K, T>>
where
    K: PositiveDefiniteKernel<T>,
    T: RandomVariable,
{
    fn ln_diff_condition(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let x_mu_mat = theta.x_mu(x)?.col_mat();
        // let x_mat = x.clone().col_mat();
        // let x_mu_mat = x_mat - mu_mat;
        let n = theta.lsigma_cols();
        let identity = DiagonalMatrix::<f64>::identity(n).mat();
        let sigma_inv = theta.sigma_inv_mul(identity).unwrap();
        println!("sigma_inv{:?}", sigma_inv);
        let sigma_inv_t = sigma_inv.t();
        println!("sigma_inv_t{:?}", sigma_inv_t);
        let f_mu = 0.5 * ((sigma_inv + sigma_inv_t.clone()) * x_mu_mat.clone());

        let x_mu_t = x_mu_mat.t();
        println!("x_mu_t{:?}", x_mu_t);
        let sigma_inv_mul_x_mu = theta.sigma_inv_mul(x_mu_t.clone()).unwrap();
        let sigma_inv_mul_sigma_inv_mul_x_mu = theta.sigma_inv_mul(sigma_inv_mul_x_mu).unwrap();
        println!(
            "sigma_inv_mul_sigma_inv_mul_x_mu{:?}",
            sigma_inv_mul_sigma_inv_mul_x_mu
        );
        let x_mul_sigma = x_mu_mat * sigma_inv_mul_sigma_inv_mul_x_mu;
        println!("x_mul_sigma{:?}", x_mul_sigma);

        let lsigma = theta.lsigma.clone().0;

        let f_sigma = (x_mul_sigma - sigma_inv_t) * lsigma;
        println!("f_sigma{:?}", f_sigma);
        let f_mu_vec = f_mu.vec();
        let f_sigma_vec = f_sigma.vec();
        let result_orig = [f_mu_vec, f_sigma_vec];
        let result = result_orig.concat();

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ConditionDifferentiableDistribution, Distribution, ExactMultivariateNormalParams,
        MultivariateNormal, SampleableDistribution, ValueDifferentiableDistribution,
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

        let f = normal
            .ln_diff_value(
                &x,
                &ExactMultivariateNormalParams::new(mu, PPTRF(lsigma)).unwrap(),
            )
            .unwrap();
        println!("{:#?}", f);
    }

    #[test]
    fn it_works_3() {
        let normal = MultivariateNormal::new();
        let mut _rng = StdRng::from_seed([1; 32]);

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

        let x = vec![0.0, 1.0, 0.0, 1.0, 2.0, 3.0];

        println!("{:#?}", lsigma);

        // let mu = vec![0.5, 1.0];
        // let lsigma = SymmetricPackedMatrix::from_mat(&mat!(
        //    1.0,  0.0;
        //    2.0,  1.0
        // ))
        // .unwrap();

        // let x = vec![0.0, 8.0];

        // let mu = vec![0.0];
        // let lsigma = SymmetricPackedMatrix::from_mat(&mat!(5.0)).unwrap();

        // let x = vec![1.0];

        // let p = normal
        //     .p_kernel(
        //         &x,
        //         &ExactMultivariateNormalParams::new(mu.clone(), PPTRF(lsigma.clone())).unwrap(),
        //     )
        //     .unwrap();
        // println!("{:#?}", p);

        let f = normal
            .ln_diff_condition(
                &x,
                &ExactMultivariateNormalParams::new(mu, PPTRF(lsigma)).unwrap(),
            )
            .unwrap();
        println!("{:#?}", f);
    }
}
