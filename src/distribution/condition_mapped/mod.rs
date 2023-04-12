// pub mod differentiable;

// pub use differentiable::*;

// use crate::{
//     DependentJoint, Distribution, DistributionError, Event, IndependentJoint, RandomVariable,
//     SamplableDistribution, ValueDifferentiableDistribution,
// };
// use rand::prelude::*;
// use std::{
//     fmt::Debug,
//     marker::PhantomData,
//     ops::{BitAnd, Mul},
// };

// #[derive(Clone)]
// pub struct ConditionMappedDistribution<D, T, U1, U2, F>
// where
//     D: Distribution<Value = T, Condition = U1>,
//     T: RandomVariable,
//     U1: Event,
//     U2: Event,
//     F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
// {
//     distribution: D,
//     condition: F,
//     phantom: PhantomData<U2>,
// }

// impl<D, T, U1, U2, F> ConditionMappedDistribution<D, T, U1, U2, F>
// where
//     D: Distribution<Value = T, Condition = U1>,
//     T: RandomVariable,
//     U1: Event,
//     U2: Event,
//     F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
// {
//     pub fn new(distribution: D, condition: F) -> Self {
//         Self {
//             distribution,
//             condition,
//             phantom: PhantomData,
//         }
//     }
// }

// impl<D, T, U1, U2, F> Debug for ConditionMappedDistribution<D, T, U1, U2, F>
// where
//     D: Distribution<Value = T, Condition = U1>,
//     T: RandomVariable,
//     U1: Event,
//     U2: Event,
//     F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
// {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(
//             f,
//             "ConditionedDistribution {{ distribution: {:#?} }}",
//             self.distribution
//         )
//     }
// }

// impl<D, T, U1, U2, F> Distribution for ConditionMappedDistribution<D, T, U1, U2, F>
// where
//     D: Distribution<Value = T, Condition = U1>,
//     T: RandomVariable,
//     U1: Event,
//     U2: Event,
//     F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
// {
//     type Value = T;
//     type Condition = U2;

//     fn p_kernel(
//         &self,
//         x: &Self::Value,
//         theta: &Self::Condition,
//     ) -> Result<f64, crate::DistributionError> {
//         self.distribution.p_kernel(x, &(self.condition)(theta)?)
//     }
// }

// pub trait ConditionMappableDistribution: Distribution + Sized {
//     /// .
//     ///
//     /// # Examples
//     ///
//     /// ```
//     /// // Example template not implemented for trait functions
//     /// ```
//     fn map_condition<U2, F>(
//         self,
//         condition: F,
//     ) -> ConditionMappedDistribution<Self, Self::Value, Self::Condition, U2, F>
//     where
//         U2: Event,
//         F: Fn(&U2) -> Result<Self::Condition, DistributionError> + Clone + Send + Sync;
// }

// impl<D, T, U1> ConditionMappableDistribution for D
// where
//     D: Distribution<Value = T, Condition = U1>,
//     T: RandomVariable,
//     U1: Event,
// {
//     fn map_condition<U2, F>(
//         self,
//         condition: F,
//     ) -> ConditionMappedDistribution<Self, Self::Value, Self::Condition, U2, F>
//     where
//         U2: Event,
//         F: Fn(&U2) -> Result<Self::Condition, DistributionError> + Clone + Send + Sync,
//     {
//         ConditionMappedDistribution::<Self, Self::Value, Self::Condition, U2, F>::new(
//             self, condition,
//         )
//     }
// }

// impl<D, T, U1, U2, Rhs, TRhs, F> Mul<Rhs> for ConditionMappedDistribution<D, T, U1, U2, F>
// where
//     D: Distribution<Value = T, Condition = U1>,
//     T: RandomVariable,
//     U1: Event,
//     U2: Event,
//     Rhs: Distribution<Value = TRhs, Condition = U2>,
//     TRhs: RandomVariable,
//     F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
// {
//     type Output = IndependentJoint<Self, Rhs, T, TRhs, U2>;

//     fn mul(self, rhs: Rhs) -> Self::Output {
//         IndependentJoint::new(self, rhs)
//     }
// }

// impl<D, T, U1, U2, Rhs, URhs, F> BitAnd<Rhs> for ConditionMappedDistribution<D, T, U1, U2, F>
// where
//     D: Distribution<Value = T, Condition = U1>,
//     T: RandomVariable,
//     U1: Event,
//     U2: Event,
//     Rhs: Distribution<Value = U2, Condition = URhs>,
//     URhs: RandomVariable,
//     F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
// {
//     type Output = DependentJoint<Self, Rhs, T, U2, URhs>;

//     fn bitand(self, rhs: Rhs) -> Self::Output {
//         DependentJoint::new(self, rhs)
//     }
// }

// impl<D, T, U1, U2, F> ValueDifferentiableDistribution
//     for ConditionMappedDistribution<D, T, U1, U2, F>
// where
//     D: Distribution<Value = T, Condition = U1> + ValueDifferentiableDistribution,
//     T: RandomVariable,
//     U1: Event,
//     U2: Event,
//     F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
// {
//     fn ln_diff_value(
//         &self,
//         x: &Self::Value,
//         theta: &Self::Condition,
//     ) -> Result<Vec<f64>, DistributionError> {
//         let f = self
//             .distribution
//             .ln_diff_value(x, &(self.condition)(theta)?)
//             .unwrap();
//         Ok(f)
//     }
// }

// impl<D, T, U1, U2, F> SamplableDistribution for ConditionMappedDistribution<D, T, U1, U2, F>
// where
//     D: SamplableDistribution<Value = T, Condition = U1>,
//     T: RandomVariable,
//     U1: Event,
//     U2: Event,
//     F: Fn(&U2) -> Result<U1, DistributionError> + Clone + Send + Sync,
// {
//     fn sample(
//         &self,
//         theta: &Self::Condition,
//         rng: &mut dyn RngCore,
//     ) -> Result<Self::Value, crate::DistributionError> {
//         self.distribution.sample(&(self.condition)(theta)?, rng)
//     }
// }

// #[cfg(test)]
// mod tests {
//     use crate::{
//         ConditionMappableDistribution, Distribution, ExactMultivariateNormalParams,
//         MultivariateNormal, SamplableDistribution, ValueDifferentiableDistribution,
//     };
//     use opensrdk_linear_algebra::{pp::trf::PPTRF, *};
//     use rand::prelude::*;

//     #[test]
//     fn it_works() {
//         let mut rng = StdRng::from_seed([1; 32]);

//         let mu = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
//         let lsigma = SymmetricPackedMatrix::from_mat(&mat!(
//            1.0,  0.0,  0.0,  0.0,  0.0,  0.0;
//            2.0,  3.0,  0.0,  0.0,  0.0,  0.0;
//            4.0,  5.0,  6.0,  0.0,  0.0,  0.0;
//            7.0,  8.0,  9.0, 10.0,  0.0,  0.0;
//           11.0, 12.0, 13.0, 14.0, 15.0,  0.0;
//           16.0, 17.0, 18.0, 19.0, 20.0, 21.0
//         ))
//         .unwrap();
//         println!("{:#?}", lsigma);

//         let distr = MultivariateNormal::new().map_condition(|theta: &Vec<f64>| {
//             let f_mu = mu
//                 .iter()
//                 .enumerate()
//                 .map(|(i, mu_i)| theta[i] + mu_i)
//                 .collect::<Vec<f64>>();
//             ExactMultivariateNormalParams::new(f_mu, PPTRF(lsigma.clone()))
//         });

//         let x = distr
//             .sample(&vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0], &mut rng)
//             .unwrap();

//         println!("{:#?}", x);
//     }

//     #[test]
//     fn it_works2() {
//         //let mut rng = StdRng::from_seed([1; 32]);

//         let mu = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
//         let lsigma = SymmetricPackedMatrix::from_mat(&mat!(
//            1.0,  0.0,  0.0,  0.0,  0.0,  0.0;
//            2.0,  3.0,  0.0,  0.0,  0.0,  0.0;
//            4.0,  5.0,  6.0,  0.0,  0.0,  0.0;
//            7.0,  8.0,  9.0, 10.0,  0.0,  0.0;
//           11.0, 12.0, 13.0, 14.0, 15.0,  0.0;
//           16.0, 17.0, 18.0, 19.0, 20.0, 21.0
//         ))
//         .unwrap();
//         println!("{:#?}", lsigma);

//         let distr = MultivariateNormal::new().map_condition(|theta: &Vec<f64>| {
//             let f_mu = mu
//                 .iter()
//                 .enumerate()
//                 .map(|(i, mu_i)| theta[i] + mu_i)
//                 .collect::<Vec<f64>>();
//             ExactMultivariateNormalParams::new(f_mu, PPTRF(lsigma.clone()))
//         });

//         let x = vec![2.0, 1.0, 0.0, 1.0, 3.0, 0.0];

//         let f = distr
//             .ln_diff_value(&x, &vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
//             .unwrap();

//         println!("{:#?}", f);
//     }
// }
