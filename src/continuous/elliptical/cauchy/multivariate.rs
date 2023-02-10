use crate::{
    ConditionDifferentiableDistribution, DependentJoint, Distribution, ExactEllipticalParams,
    ExactMultivariateStudentTParams, IndependentJoint, MultivariateStudentT,
    MultivariateStudentTParams, MultivariateStudentTWrapper, RandomVariable, SamplableDistribution,
    ValueDifferentiableDistribution,
};
use crate::{DistributionError, EllipticalParams};
use opensrdk_linear_algebra::Vector;
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

    fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let studentt_params = MultivariateStudentTWrapper::new(theta);

        MultivariateStudentT::new().p_kernel(x, &studentt_params)
    }

    // fn sample(
    //     &self,
    //     theta: &Self::Condition,
    //     rng: &mut dyn RngCore,
    // ) -> Result<Self::Value, DistributionError> {
    //     let studentt_params = MultivariateStudentTWrapper::new(theta);

    //     MultivariateStudentT::new().sample(&studentt_params, rng)
    // }
}

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

impl SamplableDistribution for MultivariateCauchy {
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let studentt_params_orig = MultivariateStudentTWrapper::new(theta);
        let studentt_params = ExactMultivariateStudentTParams::new(
            studentt_params_orig.nu(),
            studentt_params_orig.elliptical().mu().clone(),
            studentt_params_orig.elliptical().lsigma().clone(),
        )
        .unwrap();
        MultivariateStudentT::new().sample(&studentt_params, rng)
    }
}

impl ValueDifferentiableDistribution for MultivariateCauchy {
    fn ln_diff_value(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let x_mat = x.clone().row_mat();
        let mu_mat = theta.mu().clone().row_mat();
        let x_mu = x_mat - mu_mat;
        let x_mu_t = x_mu.t();
        let sigma_inv = theta.lsigma().clone().pptri()?.to_mat();

        let n = x.len() as f64;
        let d = (&x_mu * &sigma_inv * &x_mu_t)[(0, 0)];
        let f_x = -(1.0 + n) / (1.0 + &d) * x_mu * sigma_inv;
        Ok(f_x.vec())
    }
}

impl ConditionDifferentiableDistribution for MultivariateCauchy {
    fn ln_diff_condition(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let x_mat = x.clone().row_mat();
        let mu_mat = theta.mu().clone().row_mat();
        let x_mu = x_mat - mu_mat;
        let x_mu_t = x_mu.t();
        let sigma_inv = theta.lsigma().clone().pptri()?.to_mat();
        let n = x.len() as f64;
        let d = (&x_mu * &sigma_inv * &x_mu_t)[(0, 0)];
        let m = sigma_inv
            .clone()
            .hadamard_prod(&sigma_inv)
            .hadamard_prod(&sigma_inv);
        let f_mu = (1.0 + n) / (1.0 + &d) * (&x_mu * &sigma_inv);
        let f_lsigma = (1.0 + n) / (1.0 + &d) * (&x_mu * &m * &x_mu_t);
        Ok([f_mu.vec(), f_lsigma.vec()].concat())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ConditionDifferentiableDistribution, Distribution, ExactMultivariateCauchyParams,
        MultivariateCauchy, SamplableDistribution, ValueDifferentiableDistribution,
    };
    use opensrdk_linear_algebra::{pp::trf::PPTRF, *};
    use rand::prelude::*;
    #[test]
    fn it_works() {
        let cauchy = MultivariateCauchy::new();
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

        let x = cauchy
            .sample(
                &ExactMultivariateCauchyParams::new(mu, PPTRF(lsigma)).unwrap(),
                &mut rng,
            )
            .unwrap();

        println!("{:#?}", x);
    }

    #[test]
    fn it_works2() {
        let cauchy = MultivariateCauchy::new();

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
        let x = vec![0.0, 1.0, 2.0, 2.0, 2.0, 4.0];

        let f = cauchy.ln_diff_value(
            &x,
            &ExactMultivariateCauchyParams::new(mu, PPTRF(lsigma)).unwrap(),
        );
        println!("{:#?}", f);
    }

    #[test]
    fn it_works_3() {
        let cauchy = MultivariateCauchy::new();

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
        let x = vec![0.0, 1.0, 2.0, 2.0, 2.0, 4.0];

        let f = cauchy.ln_diff_condition(
            &x,
            &ExactMultivariateCauchyParams::new(mu, PPTRF(lsigma)).unwrap(),
        );
        println!("{:#?}", f);
    }
}
