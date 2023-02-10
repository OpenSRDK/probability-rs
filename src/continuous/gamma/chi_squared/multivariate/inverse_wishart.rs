use super::wishart::Wishart;
use crate::{
    DependentJoint, Distribution, IndependentJoint, RandomVariable, SamplableDistribution,
    WishartParams,
};
use crate::{DistributionError, InverseWishartParams};
use opensrdk_linear_algebra::pp::trf::PPTRF;
use rand::prelude::*;
use std::{ops::BitAnd, ops::Mul};

/// Inverse Wishart distribution
#[derive(Clone, Debug)]
pub struct InverseWishart;

#[derive(thiserror::Error, Debug)]
pub enum InverseWishartError {
    #[error("Dimension mismatch")]
    DimensionMismatch,
    #[error("'ν' must be >= dimension")]
    NuMustBeGTEDimension,
    #[error("Unknown error")]
    Unknown,
}

impl Distribution for InverseWishart {
    type Value = PPTRF;
    type Condition = InverseWishartParams;

    /// x must be cholesky decomposed
    fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let lpsi = theta.lpsi().0.to_mat();
        let nu = theta.nu();

        let p = x.0.dim() as f64;
        let lx = x.0.to_mat();

        Ok(lx.trdet().powf(-(nu + p + 1.0) / 2.0)
            * (-0.5 * x.clone().pptrs(&lpsi * lpsi.t())?.tr()).exp())
    }
}

impl<Rhs, TRhs> Mul<Rhs> for InverseWishart
where
    Rhs: Distribution<Value = TRhs, Condition = InverseWishartParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, PPTRF, TRhs, InverseWishartParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for InverseWishart
where
    Rhs: Distribution<Value = InverseWishartParams, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, PPTRF, InverseWishartParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

impl SamplableDistribution for InverseWishart {
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let lpsi = theta.lpsi();
        let nu = theta.nu();

        let lpsi_inv = lpsi.clone().pptri()?;
        let w = Wishart;
        let w_params = WishartParams::new(PPTRF(lpsi_inv), nu)?;

        let x = w.sample(&w_params, rng)?;
        let x_inv = x.pptri()?;

        Ok(x_inv.pptrf().unwrap())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
