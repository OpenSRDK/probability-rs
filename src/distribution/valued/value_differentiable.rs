use crate::{
    ConditionDifferentiableDistribution, DependentJoint, Distribution, DistributionError, Event,
    IndependentJoint, RandomVariable, SamplableDistribution, ValueDifferentiableDistribution,
    ValuedDistribution,
};
use opensrdk_linear_algebra::{Matrix, MatrixError, Vector};
use rand::prelude::*;
use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::{BitAnd, Mul},
};

#[derive(Clone)]
pub struct ValueDifferentiableValuedDistribution<D, T1, T2, U, F, G>
where
    D: Distribution<Value = T1, Condition = U>,
    T1: RandomVariable,
    T2: RandomVariable,
    U: Event,
    F: Fn(&T2) -> Result<T1, DistributionError> + Clone + Send + Sync,
    G: Fn(&T2) -> Matrix + Clone + Send + Sync,
{
    valued_distribution: ValuedDistribution<D, T1, T2, U, F>,
    value_diff: G,
    phantom: PhantomData<T2>,
}

impl<D, T1, T2, U, F, G> ValueDifferentiableValuedDistribution<D, T1, T2, U, F, G>
where
    D: Distribution<Value = T1, Condition = U>,
    T1: RandomVariable,
    T2: RandomVariable,
    U: Event,
    F: Fn(&T2) -> Result<T1, DistributionError> + Clone + Send + Sync,
    G: Fn(&T2) -> Matrix + Clone + Send + Sync,
{
    pub fn new(valued_distribution: ValuedDistribution<D, T1, T2, U, F>, value_diff: G) -> Self {
        Self {
            valued_distribution,
            value_diff,
            phantom: PhantomData,
        }
    }
}

impl<D, T1, T2, U, F, G> Debug for ValueDifferentiableValuedDistribution<D, T1, T2, U, F, G>
where
    D: Distribution<Value = T1, Condition = U>,
    T1: RandomVariable,
    T2: RandomVariable,
    U: Event,
    F: Fn(&T2) -> Result<T1, DistributionError> + Clone + Send + Sync,
    G: Fn(&T2) -> Matrix + Clone + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ValuedDistribution {{ distribution: {:#?} }}",
            self.valued_distribution.distribution
        )
    }
}

impl<D, T1, T2, U, F, G> Distribution for ValueDifferentiableValuedDistribution<D, T1, T2, U, F, G>
where
    D: Distribution<Value = T1, Condition = U>,
    T1: RandomVariable,
    T2: RandomVariable,
    U: Event,
    F: Fn(&T2) -> Result<T1, DistributionError> + Clone + Send + Sync,
    G: Fn(&T2) -> Matrix + Clone + Send + Sync,
{
    type Value = T2;
    type Condition = U;

    fn p_kernel(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<f64, crate::DistributionError> {
        self.valued_distribution
            .distribution
            .p_kernel(&(self.valued_distribution.value)(x)?, theta)
    }
}

impl<D, T1, T2, U, Rhs, TRhs, F, G> Mul<Rhs>
    for ValueDifferentiableValuedDistribution<D, T1, T2, U, F, G>
where
    D: Distribution<Value = T1, Condition = U>,
    T1: RandomVariable,
    T2: RandomVariable,
    U: Event,
    Rhs: Distribution<Value = TRhs, Condition = U>,
    TRhs: RandomVariable,
    F: Fn(&T2) -> Result<T1, DistributionError> + Clone + Send + Sync,
    G: Fn(&T2) -> Matrix + Clone + Send + Sync,
{
    type Output = IndependentJoint<Self, Rhs, T2, TRhs, U>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<D, T1, T2, U, Rhs, URhs, F, G> BitAnd<Rhs>
    for ValueDifferentiableValuedDistribution<D, T1, T2, U, F, G>
where
    D: Distribution<Value = T1, Condition = U>,
    T1: RandomVariable,
    T2: RandomVariable,
    U: Event,
    Rhs: Distribution<Value = U, Condition = URhs>,
    URhs: RandomVariable,
    F: Fn(&T2) -> Result<T1, DistributionError> + Clone + Send + Sync,
    G: Fn(&T2) -> Matrix + Clone + Send + Sync,
{
    type Output = DependentJoint<Self, Rhs, T2, U, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

impl<D, T1, T2, U, F, G> ValueDifferentiableDistribution
    for ValueDifferentiableValuedDistribution<D, T1, T2, U, F, G>
where
    D: Distribution<Value = T1, Condition = U> + ValueDifferentiableDistribution,
    T1: RandomVariable,
    T2: RandomVariable,
    U: Event,
    F: Fn(&T2) -> Result<T1, DistributionError> + Clone + Send + Sync,
    G: Fn(&T2) -> Matrix + Clone + Send + Sync,
{
    fn ln_diff_value(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let f = self
            .valued_distribution
            .distribution
            .ln_diff_value(&(self.valued_distribution.value)(x)?, theta)
            .unwrap();
        let g = &(self.value_diff)(x);

        let diff_mat = f.row_mat() * g.t();
        let diff = diff_mat.vec();

        Ok(diff)
    }
}

impl<D, T1, T2, U, F, G> ConditionDifferentiableDistribution
    for ValueDifferentiableValuedDistribution<D, T1, T2, U, F, G>
where
    D: Distribution<Value = T1, Condition = U> + ConditionDifferentiableDistribution,
    T1: RandomVariable,
    T2: RandomVariable,
    U: Event,
    F: Fn(&T2) -> Result<T1, DistributionError> + Clone + Send + Sync,
    G: Fn(&T2) -> Matrix + Clone + Send + Sync,
{
    fn ln_diff_condition(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let f = self
            .valued_distribution
            .distribution
            .ln_diff_condition(&(self.valued_distribution.value)(x)?, theta)
            .unwrap();
        Ok(f)
    }
}

impl<D, T1, T2, U, F, G> SamplableDistribution
    for ValueDifferentiableValuedDistribution<D, T1, T2, U, F, G>
where
    D: Distribution<Value = T1, Condition = U>,
    T1: RandomVariable,
    T2: RandomVariable,
    U: Event,
    F: Fn(&T2) -> Result<T1, DistributionError> + Clone + Send + Sync,
    G: Fn(&T2) -> Matrix + Clone + Send + Sync,
{
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, crate::DistributionError> {
        todo!()
    }
}

// #[cfg(test)]
// mod tests {
//   use crate::{
//       ConditionDifferentiableConditionedDistribution, ConditionDifferentiableDistribution,
//       ConditionableDistribution, Distribution, ExactMultivariateNormalParams, MultivariateNormal,
//       ValueDifferentiableDistribution,
//   };
//   use opensrdk_linear_algebra::{pp::trf::PPTRF, *};
//   use rand::prelude::*;

//   #[test]
//   fn it_works() {
//       let mut rng = StdRng::from_seed([1; 32]);

//       let mu = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
//       let lsigma = SymmetricPackedMatrix::from_mat(&mat!(
//          1.0,  0.0,  0.0,  0.0,  0.0,  0.0;
//          2.0,  3.0,  0.0,  0.0,  0.0,  0.0;
//          4.0,  5.0,  6.0,  0.0,  0.0,  0.0;
//          7.0,  8.0,  9.0, 10.0,  0.0,  0.0;
//         11.0, 12.0, 13.0, 14.0, 15.0,  0.0;
//         16.0, 17.0, 18.0, 19.0, 20.0, 21.0
//       ))
//       .unwrap();

//       let distr_prior = MultivariateNormal::new().condition(|theta: &Vec<f64>| {
//           let f_mu = mu
//               .iter()
//               .enumerate()
//               .map(|(i, mu_i)| theta[i] + mu_i)
//               .collect::<Vec<f64>>();
//           ExactMultivariateNormalParams::new(f_mu, PPTRF(lsigma.clone()))
//       });

//       let g = |_theta: &Vec<f64>| {
//           mat!(
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0
//           )
//       };

//       let distr = ConditionDifferentiableConditionedDistribution::new(distr_prior, g);

//       let x = distr
//           .sample(&vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0], &mut rng)
//           .unwrap();

//       println!("{:#?}", x);
//   }

//   #[test]
//   fn it_works2() {
//       let mut rng = StdRng::from_seed([1; 32]);

//       let mu = vec![0.0, 1.0];
//       let lsigma = SymmetricPackedMatrix::from_mat(&mat!(
//          1.0,  0.0;
//          2.0,  3.0
//       ))
//       .unwrap();

//       let distr_prior = MultivariateNormal::new().condition(|theta: &Vec<f64>| {
//           let f_mu = mu
//               .iter()
//               .enumerate()
//               .map(|(i, mu_i)| theta[i] + mu_i)
//               .collect::<Vec<f64>>();
//           ExactMultivariateNormalParams::new(f_mu, PPTRF(lsigma.clone()))
//       });

//       let g = |_theta: &Vec<f64>| {
//           mat!(
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0;
//              1.0,  1.0,  1.0,  1.0,  1.0,  1.0
//           )
//       };

//       let distr = ConditionDifferentiableConditionedDistribution::new(distr_prior, g);

//       let x = vec![2.0, 1.0];

//       let f = distr.ln_diff_condition(&x, &vec![1.0, 2.0]).unwrap();

//       println!("{:#?}", f);
//   }
// }
