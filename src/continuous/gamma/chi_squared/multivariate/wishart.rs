// // Already finished the implementation of "sampleable distribution".　The implement has commented out.

// use crate::{
//     DependentJoint, Distribution, IndependentJoint, RandomVariable, SamplableDistribution,
// };
// use crate::{DistributionError, WishartParams};
// use crate::{ExactMultivariateNormalParams, MultivariateNormal};
// use opensrdk_linear_algebra::pp::trf::PPTRF;
// use opensrdk_linear_algebra::*;
// use rand::prelude::*;
// use std::{ops::BitAnd, ops::Mul};

// #[derive(Clone, Debug, Serialize, Deserialize)]
// pub struct Wishart {
//     x: Expression,
//     nu: Expression,
//     w: Expression,
//     d: usize,
// }

// impl Wishart {
//     pub fn new(x: Expression, nu: Expression, w: Expression) -> MultivariateWishart {
//         if x.mathematical_sizes() != vec![Size::Many, Size::One] && x.mathematical_sizes() != vec![]
//         {
//             panic!("x must be a scalar or a 2 rank vector");
//         }
//         Wishart { x, nu, w, d }
//     }
// }

// #[derive(thiserror::Error, Debug)]
// pub enum WishartError {
//     #[error("Dimension mismatch")]
//     DimensionMismatch,
//     #[error("'n' must be >= dimension")]
//     NMustBeGTEDimension,
// }

// impl<Rhs> Mul<Rhs> for Wishart
// where
//     Rhs: ContinuousDistribution,
// {
//     type Output = JointDistribution<Self, Rhs>;

//     fn mul(self, rhs: Rhs) -> Self::Output {
//         JointDistribution::new(self, rhs)
//     }
// }

// impl ContinuousDistribution for Wishart {
//     fn value_ids(&self) -> HashSet<&str> {
//         self.x.variable_ids()
//     }

//     fn conditions(&self) -> Vec<&Expression> {
//         vec![&self.nu, &self.w]
//     }

//     fn pdf(&self) -> Expression {
//         let x = self.x.clone();
//         let nu = self.nu.clone();
//         let w = self.w.clone();

//         let pdf_expression = todo!();

//         pdf_expression
//     }

//     fn condition_ids(&self) -> HashSet<&str> {
//         self.conditions()
//             .iter()
//             .map(|v| v.variable_ids())
//             .flatten()
//             .collect::<HashSet<_>>()
//             .difference(&self.value_ids())
//             .cloned()
//             .collect()
//     }

//     fn ln_pdf(&self) -> Expression {
//         self.pdf().ln()
//     }
// }
