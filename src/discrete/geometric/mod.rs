// pub mod params;

// pub use params::*;

// use crate::{
//     DependentJoint, Distribution, IndependentJoint, RandomVariable, SamplableDistribution,
// };
// use crate::{DiscreteDistribution, DistributionError};
// use rand::prelude::*;
// use rand_distr::Geometric as RandGeometric;
// use std::{ops::BitAnd, ops::Mul};

// /// Geometric
// #[derive(Clone, Debug)]
// pub struct Geometric;

// #[derive(thiserror::Error, Debug)]
// pub enum GeometricError {
//     #[error("'p' must be probability.")]
//     PMustBeProbability,
// }

// impl Distribution for Geometric {
//     type Value = u64;
//     type Condition = GeometricParams;

//     fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
//         let p = theta.p();

//         Ok((1.0 - p).powi((x - 1) as i32) * p)
//     }
// }

// impl DiscreteDistribution for Geometric {}

// impl<Rhs, TRhs> Mul<Rhs> for Geometric
// where
//     Rhs: Distribution<Value = TRhs, Condition = GeometricParams>,
//     TRhs: RandomVariable,
// {
//     type Output = IndependentJoint<Self, Rhs, u64, TRhs, GeometricParams>;

//     fn mul(self, rhs: Rhs) -> Self::Output {
//         IndependentJoint::new(self, rhs)
//     }
// }

// impl<Rhs, URhs> BitAnd<Rhs> for Geometric
// where
//     Rhs: Distribution<Value = GeometricParams, Condition = URhs>,
//     URhs: RandomVariable,
// {
//     type Output = DependentJoint<Self, Rhs, u64, GeometricParams, URhs>;

//     fn bitand(self, rhs: Rhs) -> Self::Output {
//         DependentJoint::new(self, rhs)
//     }
// }

// impl SamplableDistribution for Geometric {
//     fn sample(
//         &self,
//         theta: &Self::Condition,
//         rng: &mut dyn RngCore,
//     ) -> Result<Self::Value, DistributionError> {
//         let p = theta.p();

//         let geometric = match RandGeometric::new(p) {
//             Ok(v) => Ok(v),
//             Err(e) => Err(DistributionError::Others(e.into())),
//         }?;

//         Ok(rng.sample(geometric))
//     }
// }
// #[cfg(test)]
// mod tests {
//     #[test]
//     fn it_works() {
//         assert_eq!(2 + 2, 4);
//     }
// }
