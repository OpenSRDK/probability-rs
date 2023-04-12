// use crate::{
//     ConditionDifferentiableDistribution, DistributionError, SamplableDistribution,
//     ValueDifferentiableDistribution,
// };
// use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
// use rand::prelude::*;
// use std::iter::Iterator;
// use std::{ops::BitAnd, ops::Mul};

// /// p(x|a) = Π p(xi|ai)
// #[derive(Clone, Debug)]
// pub struct IndependentArrayJoint<D, T, U>
// where
//     D: Distribution<Value = T, Condition = U>,
//     T: RandomVariable,
//     U: RandomVariable,
// {
//     distributions: Vec<D>,
// }

// impl<D, T, U> Distribution for IndependentArrayJoint<D, T, U>
// where
//     D: Distribution<Value = T, Condition = U>,
//     T: RandomVariable,
//     U: RandomVariable,
// {
//     type Value = Vec<T>;
//     type Condition = Vec<U>;

//     fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
//         x.iter()
//             .zip(theta.iter())
//             .enumerate()
//             .map(|(i, (xi, thetai))| self.distributions[i].p_kernel(xi, thetai))
//             .product()
//     }
// }

// impl<D, T, U, Rhs, TRhs> Mul<Rhs> for IndependentArrayJoint<D, T, U>
// where
//     D: Distribution<Value = T, Condition = U>,
//     T: RandomVariable,
//     U: RandomVariable,
//     Rhs: Distribution<Value = TRhs, Condition = Vec<U>>,
//     TRhs: RandomVariable,
// {
//     type Output = IndependentJoint<Self, Rhs, Vec<T>, TRhs, Vec<U>>;

//     fn mul(self, rhs: Rhs) -> Self::Output {
//         IndependentJoint::new(self, rhs)
//     }
// }

// impl<D, T, U, Rhs, URhs> BitAnd<Rhs> for IndependentArrayJoint<D, T, U>
// where
//     D: Distribution<Value = T, Condition = U>,
//     T: RandomVariable,
//     U: RandomVariable,
//     Rhs: Distribution<Value = Vec<U>, Condition = URhs>,
//     URhs: RandomVariable,
// {
//     type Output = DependentJoint<Self, Rhs, Vec<T>, Vec<U>, URhs>;

//     fn bitand(self, rhs: Rhs) -> Self::Output {
//         DependentJoint::new(self, rhs)
//     }
// }

// pub trait DistributionProduct<D, T, U>
// where
//     D: Distribution<Value = T, Condition = U>,
//     T: RandomVariable,
//     U: RandomVariable,
// {
//     /// p(x|a) = Π p(xi|ai)
//     fn joint(self) -> IndependentArrayJoint<D, T, U>;
// }

// impl<I, D, T, U> DistributionProduct<D, T, U> for I
// where
//     I: Iterator<Item = D>,
//     D: Distribution<Value = T, Condition = U>,
//     T: RandomVariable,
//     U: RandomVariable,
// {
//     fn joint(self) -> IndependentArrayJoint<D, T, U> {
//         let distributions = self.collect::<Vec<_>>();

//         IndependentArrayJoint::<D, T, U> { distributions }
//     }
// }

// impl<D, T, U> ValueDifferentiableDistribution for IndependentArrayJoint<D, T, U>
// where
//     D: Distribution<Value = T, Condition = U> + ValueDifferentiableDistribution,
//     T: RandomVariable,
//     U: RandomVariable,
// {
//     fn ln_diff_value(
//         &self,
//         x: &Self::Value,
//         theta: &Self::Condition,
//     ) -> Result<Vec<f64>, DistributionError> {
//         let f = x
//             .iter()
//             .enumerate()
//             .flat_map(|(i, xi)| {
//                 self.distributions[i]
//                     .ln_diff_value(xi, &theta[i])
//                     .unwrap()
//                     .into_iter()
//             })
//             .collect::<Vec<f64>>();
//         Ok(f)
//     }
// }

// impl<D, T, U> ConditionDifferentiableDistribution for IndependentArrayJoint<D, T, U>
// where
//     D: Distribution<Value = T, Condition = U> + ConditionDifferentiableDistribution,
//     T: RandomVariable,
//     U: RandomVariable,
// {
//     fn ln_diff_condition(
//         &self,
//         x: &Self::Value,
//         theta: &Self::Condition,
//     ) -> Result<Vec<f64>, DistributionError> {
//         let f = x
//             .iter()
//             .enumerate()
//             .flat_map(|(i, xi)| {
//                 self.distributions[i]
//                     .ln_diff_condition(xi, &theta[i])
//                     .unwrap()
//                     .into_iter()
//             })
//             .collect::<Vec<f64>>();
//         Ok(f)
//     }
// }

// impl<D, T, U> SamplableDistribution for IndependentArrayJoint<D, T, U>
// where
//     D: SamplableDistribution<Value = T, Condition = U>,
//     T: RandomVariable,
//     U: RandomVariable,
// {
//     fn sample(
//         &self,
//         theta: &Self::Condition,
//         rng: &mut dyn RngCore,
//     ) -> Result<Self::Value, DistributionError> {
//         self.distributions
//             .iter()
//             .zip(theta.iter())
//             .map(|(di, thetai)| di.sample(thetai, rng))
//             .collect()
//     }
// }

// #[cfg(test)]
// mod tests {
//     use crate::distribution::Distribution;
//     use crate::*;
//     use rand::prelude::*;
//     #[test]
//     fn it_works() {
//         let model = vec![Normal; 3].into_iter().joint();

//         let mut rng = StdRng::from_seed([1; 32]);

//         let x = model
//             .sample(&vec![NormalParams::new(0.0, 1.0).unwrap(); 3], &mut rng)
//             .unwrap();

//         println!("{:#?}", x);
//     }

//     #[test]
//     fn it_works2() {
//         let model = vec![Normal; 3].into_iter().joint();
//         let f = model
//             .ln_diff_value(
//                 &vec![0.0, 1.0, 2.0],
//                 &vec![NormalParams::new(0.0, 1.0).unwrap(); 3],
//             )
//             .unwrap();

//         println!("{:#?}", f);
//     }

//     #[test]
//     fn it_works3() {
//         let model = vec![Normal; 3].into_iter().joint();
//         let f = model
//             .ln_diff_condition(
//                 &vec![0.0, 1.0, 2.0],
//                 &vec![NormalParams::new(0.0, 1.0).unwrap(); 3],
//             )
//             .unwrap();

//         println!("{:#?}", f);
//     }
// }
