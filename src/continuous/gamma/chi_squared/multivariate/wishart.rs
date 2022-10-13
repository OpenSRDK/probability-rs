// Already finished the implementation of "sampleable distribution".　The implement has commented out.

use crate::{
    DependentJoint, Distribution, IndependentJoint, RandomVariable, SampleableDistribution,
};
use crate::{DistributionError, WishartParams};
use crate::{ExactMultivariateNormalParams, MultivariateNormal};
use opensrdk_linear_algebra::pp::trf::PPTRF;
use opensrdk_linear_algebra::*;
use rand::prelude::*;
use std::{ops::BitAnd, ops::Mul};

/// Wishart distribution
#[derive(Clone, Debug)]
pub struct Wishart;

#[derive(thiserror::Error, Debug)]
pub enum WishartError {
    #[error("Dimension mismatch")]
    DimensionMismatch,
    #[error("'n' must be >= dimension")]
    NMustBeGTEDimension,
}

impl Distribution for Wishart {
    type Value = PPTRF;
    type Condition = WishartParams;

    /// x must be cholesky decomposed
    fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let lv = theta.lv();
        let n = theta.n();

        let p = x.0.dim() as f64;
        let lx = x.0.to_mat();

        Ok(lx.trdet().powf(n + p + 1.0) * (-0.5 * lv.clone().pptrs(&lx * lx.t())?.tr()).exp())
    }
}

impl<Rhs, TRhs> Mul<Rhs> for Wishart
where
    Rhs: Distribution<Value = TRhs, Condition = WishartParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, PPTRF, TRhs, WishartParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for Wishart
where
    Rhs: Distribution<Value = WishartParams, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, PPTRF, WishartParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

impl SampleableDistribution for Wishart {
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let lv = theta.lv();
        let n = theta.n() as usize;

        let p = lv.0.dim();

        let normal = MultivariateNormal::new();
        let normal_params = ExactMultivariateNormalParams::new(vec![0.0; p], lv.clone())?;

        let w = (0..n)
            .into_iter()
            .map(|_| normal.sample(&normal_params, rng))
            .try_fold::<Matrix, _, Result<Matrix, DistributionError>>(
                Matrix::new(p, p),
                |acc, v: Result<Vec<f64>, DistributionError>| {
                    let v = v?;
                    Ok(acc + v.clone().row_mat() * v.col_mat())
                },
            )?;

        Ok(SymmetricPackedMatrix::from_mat(&w)
            .unwrap()
            .pptrf()
            .unwrap())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
