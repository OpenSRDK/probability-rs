// pub mod value_differentiable;

// pub use value_differentiable::*;

// use crate::{
//     ConditionDifferentiableDistribution, DependentJoint, Distribution, DistributionError, Event,
//     IndependentJoint, RandomVariable, SamplableDistribution,
// };
// use rand::prelude::*;
// use std::{
//     fmt::Debug,
//     marker::PhantomData,
//     ops::{BitAnd, Mul},
// };

// #[derive(Clone)]
// pub struct ValuedDistribution<D, T1, T2, U, F>
// where
//     D: Distribution<Value = T1, Condition = U>,
//     T1: RandomVariable,
//     T2: RandomVariable,
//     U: Event,
//     F: Fn(&T2) -> Result<T1, DistributionError> + Clone + Send + Sync,
// {
//     distribution: D,
//     value: F,
//     phantom: PhantomData<T2>,
// }

// impl<D, T1, T2, U, F> ValuedDistribution<D, T1, T2, U, F>
// where
//     D: Distribution<Value = T1, Condition = U>,
//     T1: RandomVariable,
//     T2: RandomVariable,
//     U: Event,
//     F: Fn(&T2) -> Result<T1, DistributionError> + Clone + Send + Sync,
// {
//     pub fn new(distribution: D, value: F) -> Self {
//         Self {
//             distribution,
//             value,
//             phantom: PhantomData,
//         }
//     }
// }

// impl<D, T1, T2, U, F> Debug for ValuedDistribution<D, T1, T2, U, F>
// where
//     D: Distribution<Value = T1, Condition = U>,
//     T1: RandomVariable,
//     T2: RandomVariable,
//     U: Event,
//     F: Fn(&T2) -> Result<T1, DistributionError> + Clone + Send + Sync,
// {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(
//             f,
//             "ValuedDistribution {{ distribution: {:#?} }}",
//             self.distribution
//         )
//     }
// }

// impl<D, T1, T2, U, F> Distribution for ValuedDistribution<D, T1, T2, U, F>
// where
//     D: Distribution<Value = T1, Condition = U>,
//     T1: RandomVariable,
//     T2: RandomVariable,
//     U: Event,
//     F: Fn(&T2) -> Result<T1, DistributionError> + Clone + Send + Sync,
// {
//     type Value = T2;
//     type Condition = U;

//     fn p_kernel(
//         &self,
//         x: &Self::Value,
//         theta: &Self::Condition,
//     ) -> Result<f64, crate::DistributionError> {
//         self.distribution.p_kernel(&(self.value)(x)?, theta)
//     }
// }

// pub trait ValuableDistribution: Distribution + Sized {
//     /// .
//     ///
//     /// # Examples
//     ///
//     /// ```
//     /// // Example template not implemented for trait functions
//     /// ```
//     fn value<T2, F>(
//         self,
//         value: F,
//     ) -> ValuedDistribution<Self, Self::Value, T2, Self::Condition, F>
//     where
//         T2: RandomVariable,
//         F: Fn(&T2) -> Result<Self::Value, DistributionError> + Clone + Send + Sync;
// }

// impl<D, T1, U> ValuableDistribution for D
// where
//     D: Distribution<Value = T1, Condition = U>,
//     T1: RandomVariable,
//     U: Event,
// {
//     fn value<T2, F>(self, value: F) -> ValuedDistribution<Self, Self::Value, T2, Self::Condition, F>
//     where
//         T2: RandomVariable,
//         F: Fn(&T2) -> Result<Self::Value, DistributionError> + Clone + Send + Sync,
//     {
//         ValuedDistribution::<Self, Self::Value, T2, Self::Condition, F>::new(self, value)
//     }
// }

// impl<D, T1, T2, U, Rhs, TRhs, F> Mul<Rhs> for ValuedDistribution<D, T1, T2, U, F>
// where
//     D: Distribution<Value = T1, Condition = U>,
//     T1: RandomVariable,
//     T2: RandomVariable,
//     U: Event,
//     Rhs: Distribution<Value = TRhs, Condition = U>,
//     TRhs: RandomVariable,
//     F: Fn(&T2) -> Result<T1, DistributionError> + Clone + Send + Sync,
// {
//     type Output = IndependentJoint<Self, Rhs, T2, TRhs, U>;

//     fn mul(self, rhs: Rhs) -> Self::Output {
//         IndependentJoint::new(self, rhs)
//     }
// }

// impl<D, T1, T2, U, Rhs, URhs, F> BitAnd<Rhs> for ValuedDistribution<D, T1, T2, U, F>
// where
//     D: Distribution<Value = T1, Condition = U>,
//     T1: RandomVariable,
//     T2: RandomVariable,
//     U: Event,
//     Rhs: Distribution<Value = U, Condition = URhs>,
//     URhs: RandomVariable,
//     F: Fn(&T2) -> Result<T1, DistributionError> + Clone + Send + Sync,
// {
//     type Output = DependentJoint<Self, Rhs, T2, U, URhs>;

//     fn bitand(self, rhs: Rhs) -> Self::Output {
//         DependentJoint::new(self, rhs)
//     }
// }

// impl<D, T1, T2, U, F> ConditionDifferentiableDistribution for ValuedDistribution<D, T1, T2, U, F>
// where
//     D: Distribution<Value = T1, Condition = U> + ConditionDifferentiableDistribution,
//     T1: RandomVariable,
//     T2: RandomVariable,
//     U: Event,
//     F: Fn(&T2) -> Result<T1, DistributionError> + Clone + Send + Sync,
// {
//     fn ln_diff_condition(
//         &self,
//         x: &Self::Value,
//         theta: &Self::Condition,
//     ) -> Result<Vec<f64>, DistributionError> {
//         let f = self
//             .distribution
//             .ln_diff_condition(&(self.value)(x)?, theta)
//             .unwrap();
//         Ok(f)
//     }
// }

// impl<D, T1, T2, U, F> SamplableDistribution for ValuedDistribution<D, T1, T2, U, F>
// where
//     D: SamplableDistribution<Value = T1, Condition = U>,
//     T1: RandomVariable,
//     T2: RandomVariable,
//     U: Event,
//     F: Fn(&T2) -> Result<T1, DistributionError> + Clone + Send + Sync,
// {
//     fn sample(
//         &self,
//         theta: &Self::Condition,
//         rng: &mut dyn RngCore,
//     ) -> Result<Self::Value, crate::DistributionError> {
//         todo!()
//     }
// }

// // #[cfg(test)]
// // mod tests {
// //     use crate::{
// //         ConditionDifferentiableDistribution, ConditionableDistribution, Distribution,
// //         ExactMultivariateNormalParams, MultivariateNormal, ValueDifferentiableDistribution,
// //     };
// //     use opensrdk_linear_algebra::{pp::trf::PPTRF, *};
// //     use rand::prelude::*;

// //     #[test]
// //     fn it_works() {
// //         let mut rng = StdRng::from_seed([1; 32]);

// //         let mu = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
// //         let lsigma = SymmetricPackedMatrix::from_mat(&mat!(
// //            1.0,  0.0,  0.0,  0.0,  0.0,  0.0;
// //            2.0,  3.0,  0.0,  0.0,  0.0,  0.0;
// //            4.0,  5.0,  6.0,  0.0,  0.0,  0.0;
// //            7.0,  8.0,  9.0, 10.0,  0.0,  0.0;
// //           11.0, 12.0, 13.0, 14.0, 15.0,  0.0;
// //           16.0, 17.0, 18.0, 19.0, 20.0, 21.0
// //         ))
// //         .unwrap();
// //         println!("{:#?}", lsigma);

// //         let distr = MultivariateNormal::new().condition(|theta: &Vec<f64>| {
// //             let f_mu = mu
// //                 .iter()
// //                 .enumerate()
// //                 .map(|(i, mu_i)| theta[i] + mu_i)
// //                 .collect::<Vec<f64>>();
// //             ExactMultivariateNormalParams::new(f_mu, PPTRF(lsigma.clone()))
// //         });

// //         let x = distr
// //             .sample(&vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0], &mut rng)
// //             .unwrap();

// //         println!("{:#?}", x);
// //     }

// //     #[test]
// //     fn it_works2() {
// //         let mut rng = StdRng::from_seed([1; 32]);

// //         let mu = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
// //         let lsigma = SymmetricPackedMatrix::from_mat(&mat!(
// //            1.0,  0.0,  0.0,  0.0,  0.0,  0.0;
// //            2.0,  3.0,  0.0,  0.0,  0.0,  0.0;
// //            4.0,  5.0,  6.0,  0.0,  0.0,  0.0;
// //            7.0,  8.0,  9.0, 10.0,  0.0,  0.0;
// //           11.0, 12.0, 13.0, 14.0, 15.0,  0.0;
// //           16.0, 17.0, 18.0, 19.0, 20.0, 21.0
// //         ))
// //         .unwrap();
// //         println!("{:#?}", lsigma);

// //         let distr = MultivariateNormal::new().condition(|theta: &Vec<f64>| {
// //             let f_mu = mu
// //                 .iter()
// //                 .enumerate()
// //                 .map(|(i, mu_i)| theta[i] + mu_i)
// //                 .collect::<Vec<f64>>();
// //             ExactMultivariateNormalParams::new(f_mu, PPTRF(lsigma.clone()))
// //         });

// //         let x = vec![2.0, 1.0, 0.0, 1.0, 3.0, 0.0];

// //         let f = distr
// //             .ln_diff_value(&x, &vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
// //             .unwrap();

// //         println!("{:#?}", f);
// //     }
// // }
