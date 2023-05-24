// // Already finished the implementation of "sampleable distribution".　The implement has commented out.

pub mod chi_squared;
pub mod params;

use std::collections::HashSet;

pub use chi_squared::*;
use opensrdk_symbolic_computation::{ConstantValue, Expression};
pub use params::*;
use rand_distr::Gamma as RandGamma;
use serde::{Deserialize, Serialize};

use crate::ContinuousDistribution;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Gamma {
    x: Expression,
    shape: Expression,
    scale: Expression,
}

impl ContinuousDistribution for Gamma {
    fn value_ids(&self) -> HashSet<&str> {
        self.x.variable_ids()
    }

    fn conditions(&self) -> Vec<&Expression> {
        vec![&self.shape, &self.scale]
    }

    fn pdf(&self) -> Expression {
        let x = self.x.clone();
        let shape = self.shape.clone();
        let shapeValue: f64;
        let scale = self.scale.clone();
        let scaleValue: f64;

        if let Expression::Constant(value) = shape {
            let constantValue: ConstantValue = value;
            let shapeValue = constantValue.into_scalar();
        } else {
            panic!("shape must be a constant value");
        }

        if let Expression::Constant(value) = scale {
            let constantValue: ConstantValue = value;
            let scaleValue = constantValue.into_scalar();
        } else {
            panic!("scale must be a constant value");
        }

        // let gamma = match RandGamma::new(shapeValue, scaleValue) {
        //     Ok(v) => Ok(v),
        //     Err(e) => Err(e.into()),
        // }?;

        // let pdf = todo!();
        todo!()
    }

    fn condition_ids(&self) -> HashSet<&str> {
        self.conditions()
            .iter()
            .map(|v| v.variable_ids())
            .flatten()
            .collect::<HashSet<_>>()
            .difference(&self.value_ids())
            .cloned()
            .collect()
    }

    fn ln_pdf(&self) -> Expression {
        self.pdf().ln()
    }
}

// use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
// use crate::{DistributionError, SamplableDistribution};
// use rand::prelude::*;
// use std::{ops::BitAnd, ops::Mul};

// /// Gamma distribution
// #[derive(Clone, Debug)]
// pub struct Gamma;

// #[derive(thiserror::Error, Debug)]
// pub enum GammaError {
//     #[error("'shape' must be positive")]
//     ShapeMustBePositive,
//     #[error("'scale' must be positive")]
//     ScaleMustBePositive,
// }

// impl Distribution for Gamma {
//     type Value = f64;
//     type Condition = GammaParams;

//     fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
//         let shape = theta.shape();
//         let scale = theta.scale();

//         Ok(x.powf(shape - 1.0) * (-x / scale).exp())
//     }
// }

// impl<Rhs, TRhs> Mul<Rhs> for Gamma
// where
//     Rhs: Distribution<Value = TRhs, Condition = GammaParams>,
//     TRhs: RandomVariable,
// {
//     type Output = IndependentJoint<Self, Rhs, f64, TRhs, GammaParams>;

//     fn mul(self, rhs: Rhs) -> Self::Output {
//         IndependentJoint::new(self, rhs)
//     }
// }

// impl<Rhs, URhs> BitAnd<Rhs> for Gamma
// where
//     Rhs: Distribution<Value = GammaParams, Condition = URhs>,
//     URhs: RandomVariable,
// {
//     type Output = DependentJoint<Self, Rhs, f64, GammaParams, URhs>;

//     fn bitand(self, rhs: Rhs) -> Self::Output {
//         DependentJoint::new(self, rhs)
//     }
// }

// impl SamplableDistribution for Gamma {
//     fn sample(
//         &self,
//         theta: &Self::Condition,
//         rng: &mut dyn RngCore,
//     ) -> Result<Self::Value, DistributionError> {
//         let shape = theta.shape();
//         let scale = theta.scale();

//         let gamma = match RandGamma::new(shape, scale) {
//             Ok(v) => Ok(v),
//             Err(e) => Err(DistributionError::Others(e.into())),
//         }?;

//         Ok(rng.sample(gamma))
//     }
// }

// #[cfg(test)]
// mod tests {
//     #[test]
//     fn it_works() {
//         assert_eq!(2 + 2, 4);
//     }
// }
