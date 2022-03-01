use crate::{
    ConditionDifferentiableDistribution, DependentJoint, Distribution, IndependentJoint,
    RandomVariable,
};
use crate::{DistributionError, ValueDifferentiableDistribution};
use opensrdk_linear_algebra::{mat, Vector};
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

    fn fk(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        x.iter()
            .enumerate()
            .map(|(i, xi)| self.distributions[i].fk(xi, theta))
            .product()
    }

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
            .fold(mat!(0.0, 0.0, 0.0), |a, b| a + b)
            .vec();
        Ok(f)
    }
}

#[cfg(test)]
mod tests {
    use crate::distribution::Distribution;
    use crate::*;
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
}
