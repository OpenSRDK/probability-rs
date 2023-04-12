// use crate::{
//     DependentJoint, Distribution, IndependentJoint, RandomVariable, SamplableDistribution,
// };
// use crate::{DistributionError, Event};
// use rand::prelude::*;
// use std::{
//     collections::HashMap,
//     fmt::Debug,
//     ops::{BitAnd, Mul},
// };

// pub mod params;

// pub use params::*;

// #[derive(Clone, Debug)]
// pub struct SwitchedDistribution<'a, D, T, U>
// where
//     D: Distribution<Value = T, Condition = U>,
//     T: RandomVariable,
//     U: Clone + Debug + Send + Sync,
// {
//     distribution: &'a D,
//     map: &'a HashMap<u32, U>,
// }

// #[derive(thiserror::Error, Debug)]
// pub enum SwitchedError {
//     #[error("Key not found")]
//     KeyNotFound,
//     #[error("Unknown error")]
//     Unknown,
// }

// impl<'a, D, T, U> SwitchedDistribution<'a, D, T, U>
// where
//     D: Distribution<Value = T, Condition = U>,
//     T: RandomVariable,
//     U: Event,
// {
//     pub fn new(distribution: &'a D, map: &'a HashMap<u32, U>) -> Self {
//         Self { distribution, map }
//     }

//     pub fn distribution(&self) -> &D {
//         &self.distribution
//     }
// }

// impl<'a, D, T, U> Distribution for SwitchedDistribution<'a, D, T, U>
// where
//     D: Distribution<Value = T, Condition = U>,
//     T: RandomVariable,
//     U: Event,
// {
//     type Value = T;
//     type Condition = SwitchedParams<U>;

//     fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
//         let s = theta;

//         match s {
//             SwitchedParams::Key(k) => match self.map.get(k) {
//                 Some(theta) => self.distribution.p_kernel(x, theta),
//                 None => Err(DistributionError::InvalidParameters(
//                     SwitchedError::KeyNotFound.into(),
//                 )),
//             },
//             SwitchedParams::Direct(theta) => self.distribution.p_kernel(x, theta),
//         }
//     }
// }

// pub trait SwitchableDistribution<U>: Distribution + Sized
// where
//     U: Event,
// {
//     fn switch<'a>(
//         &'a self,
//         map: &'a HashMap<u32, U>,
//     ) -> SwitchedDistribution<'a, Self, Self::Value, Self::Condition>;
// }

// impl<D, T, U> SwitchableDistribution<U> for D
// where
//     D: Distribution<Value = T, Condition = U>,
//     T: RandomVariable,
//     U: Event,
// {
//     fn switch<'a>(
//         &'a self,
//         map: &'a HashMap<u32, U>,
//     ) -> SwitchedDistribution<'a, Self, Self::Value, U> {
//         SwitchedDistribution::<Self, Self::Value, U>::new(self, map)
//     }
// }

// impl<'a, D, T, U, Rhs, TRhs> Mul<Rhs> for SwitchedDistribution<'a, D, T, U>
// where
//     D: Distribution<Value = T, Condition = U>,
//     T: RandomVariable,
//     U: Event,
//     Rhs: Distribution<Value = TRhs, Condition = SwitchedParams<U>>,
//     TRhs: RandomVariable,
// {
//     type Output = IndependentJoint<Self, Rhs, T, TRhs, SwitchedParams<U>>;

//     fn mul(self, rhs: Rhs) -> Self::Output {
//         IndependentJoint::new(self, rhs)
//     }
// }

// impl<'a, D, T, U, Rhs, URhs> BitAnd<Rhs> for SwitchedDistribution<'a, D, T, U>
// where
//     D: Distribution<Value = T, Condition = U>,
//     T: RandomVariable,
//     U: Event,
//     Rhs: Distribution<Value = SwitchedParams<U>, Condition = URhs>,
//     URhs: Event,
// {
//     type Output = DependentJoint<Self, Rhs, T, SwitchedParams<U>, URhs>;

//     fn bitand(self, rhs: Rhs) -> Self::Output {
//         DependentJoint::new(self, rhs)
//     }
// }

// impl<'a, D, T, U> SamplableDistribution for SwitchedDistribution<'a, D, T, U>
// where
//     D: SamplableDistribution<Value = T, Condition = U>,
//     T: RandomVariable,
//     U: Event,
// {
//     fn sample(
//         &self,
//         theta: &Self::Condition,
//         rng: &mut dyn RngCore,
//     ) -> Result<Self::Value, DistributionError> {
//         let s = theta;

//         match s {
//             SwitchedParams::Key(k) => match self.map.get(k) {
//                 Some(theta) => self.distribution.sample(theta, rng),
//                 None => Err(DistributionError::InvalidParameters(
//                     SwitchedError::KeyNotFound.into(),
//                 )),
//             },
//             SwitchedParams::Direct(theta) => self.distribution.sample(theta, rng),
//         }
//     }
// }

// #[cfg(test)]
// mod tests {
//     use crate::*;
//     use std::collections::HashMap;

//     #[test]
//     fn it_works() {
//         let mut theta = HashMap::new();
//         theta.insert(1u32, NormalParams::new(1.0, 2.0).unwrap());
//         theta.insert(2u32, NormalParams::new(2.0, 2.0).unwrap());
//         theta.insert(3u32, NormalParams::new(3.0, 2.0).unwrap());
//         theta.insert(4u32, NormalParams::new(4.0, 2.0).unwrap());
//         let distr = Normal.switch(&theta);
//         let switched_fk = distr.p_kernel(&0f64, &SwitchedParams::Key(1u32)).unwrap();
//         let fk = Normal
//             .p_kernel(&0f64, &NormalParams::new(1.0, 2.0).unwrap())
//             .unwrap();

//         assert_eq!(switched_fk, fk);
//     }
// }
