use crate::{
    ConditionDifferentiableDistribution, DependentJoint, Distribution, IndependentJoint,
    RandomVariable, SampleableDistribution,
};
use crate::{DistributionError, ValueDifferentiableDistribution};
use opensrdk_linear_algebra::Vector;
use rand::prelude::*;
use std::iter::Iterator;
use std::{ops::BitAnd, ops::Mul};

/// p(x|a) = Π p(xi|a)
#[derive(Clone, Debug)]
pub struct IndependentValueArrayJoint<D, T, U>
where
    D: Distribution<Value = T, Condition = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    distributions: Vec<D>,
}

impl<D, T, U> Distribution for IndependentValueArrayJoint<D, T, U>
where
    D: Distribution<Value = T, Condition = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    type Value = Vec<T>;
    type Condition = U;

    fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        x.iter()
            .enumerate()
            .map(|(i, xi)| self.distributions[i].p_kernel(xi, theta))
            .product()
    }
}

impl<D, T, U, Rhs, TRhs> Mul<Rhs> for IndependentValueArrayJoint<D, T, U>
where
    D: Distribution<Value = T, Condition = U>,
    T: RandomVariable,
    U: RandomVariable,
    Rhs: Distribution<Value = TRhs, Condition = U>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, Vec<T>, TRhs, U>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<D, T, U, Rhs, URhs> BitAnd<Rhs> for IndependentValueArrayJoint<D, T, U>
where
    D: Distribution<Value = T, Condition = U>,
    T: RandomVariable,
    U: RandomVariable,
    Rhs: Distribution<Value = U, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, Vec<T>, U, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

pub trait DistributionValueProduct<D, T, U>
where
    D: Distribution<Value = T, Condition = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    fn only_value_joint(self) -> IndependentValueArrayJoint<D, T, U>;
}

impl<I, D, T, U> DistributionValueProduct<D, T, U> for I
where
    I: Iterator<Item = D>,
    D: Distribution<Value = T, Condition = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    fn only_value_joint(self) -> IndependentValueArrayJoint<D, T, U> {
        let distributions = self.collect::<Vec<_>>();

        IndependentValueArrayJoint::<D, T, U> { distributions }
    }
}

impl<D, T, U> ValueDifferentiableDistribution for IndependentValueArrayJoint<D, T, U>
where
    D: Distribution<Value = T, Condition = U> + ValueDifferentiableDistribution,
    T: RandomVariable,
    U: RandomVariable,
{
    fn ln_diff_value(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let f = x
            .iter()
            .enumerate()
            .flat_map(|(i, xi)| {
                self.distributions[i]
                    .ln_diff_value(xi, theta)
                    .unwrap()
                    .into_iter()
            })
            .collect::<Vec<f64>>();
        Ok(f)
    }
}

impl<D, T, U> ConditionDifferentiableDistribution for IndependentValueArrayJoint<D, T, U>
where
    D: Distribution<Value = T, Condition = U> + ConditionDifferentiableDistribution,
    T: RandomVariable,
    U: RandomVariable,
{
    fn ln_diff_condition(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let f = x
            .iter()
            .enumerate()
            .map(|(i, xi)| {
                self.distributions[i]
                    .ln_diff_condition(xi, theta)
                    .unwrap()
                    .col_mat()
            })
            .fold(
                vec![
                    0.0;
                    self.distributions[0]
                        .ln_diff_condition(&x[0], theta)
                        .unwrap()
                        .col_mat()
                        .len()
                ]
                .col_mat(),
                |a, b| a + b,
            )
            .vec();
        Ok(f)
    }
}

impl<D, T, U> SampleableDistribution for IndependentValueArrayJoint<D, T, U>
where
    D: SampleableDistribution<Value = T, Condition = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        self.distributions
            .iter()
            .map(|di| di.sample(theta, rng))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::distribution::Distribution;
    use crate::*;
    use opensrdk_linear_algebra::{mat, pp::trf::PPTRF, SymmetricPackedMatrix};
    use rand::prelude::*;
    #[test]
    fn it_works() {
        let model = vec![Normal; 3].into_iter().only_value_joint();

        let mut rng = StdRng::from_seed([1; 32]);

        let x = model
            .sample(&NormalParams::new(0.0, 1.0).unwrap(), &mut rng)
            .unwrap();

        println!("{:#?}", x);
    }

    #[test]
    fn it_works2() {
        let model = vec![Normal; 3].into_iter().only_value_joint();

        let f = model
            .ln_diff_value(&vec![1.0, 2.0, 3.0], &NormalParams::new(0.0, 1.0).unwrap())
            .unwrap();

        println!("{:#?}", f);
    }

    #[test]
    fn it_works3() {
        let model = vec![Normal; 3].into_iter().only_value_joint();

        let f = model
            .ln_diff_condition(&vec![1.0, 2.0, 3.0], &NormalParams::new(0.0, 1.0).unwrap())
            .unwrap();

        println!("{:#?}", f);
    }

    #[test]
    fn it_works4() {
        let mu = vec![0.0, 1.0];
        let lsigma = SymmetricPackedMatrix::from_mat(&mat!(
           1.0,  0.0;
           2.0,  3.0
        ))
        .unwrap();

        let model = vec![MultivariateNormal::new(); 3]
            .into_iter()
            .only_value_joint();

        let f = model
            .ln_diff_condition(
                &vec![vec![2.0, 1.0], vec![0.0, 1.0], vec![2.0, 0.0]],
                &ExactMultivariateNormalParams::new(mu, PPTRF(lsigma)).unwrap(),
            )
            .unwrap();

        println!("{:#?}", f);
    }
}
