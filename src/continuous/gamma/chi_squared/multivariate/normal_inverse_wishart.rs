use crate::{
    DependentJoint, Distribution, ExactMultivariateNormalParams, IndependentJoint, InverseWishart,
    InverseWishartParams, MultivariateNormal, RandomVariable,
};
use crate::{DistributionError, NormalInverseWishartParams};
use opensrdk_linear_algebra::pp::trf::PPTRF;
use opensrdk_linear_algebra::{SymmetricPackedMatrix, Vector};
use rand::prelude::*;
use std::{ops::BitAnd, ops::Mul};

/// Normal inverse Wishart distribution
#[derive(Clone, Debug)]
pub struct NormalInverseWishart;

#[derive(thiserror::Error, Debug)]
pub enum NormalInverseWishartError {
    #[error("Dimension mismatch")]
    DimensionMismatch,
    #[error("'λ' must be positive")]
    LambdaMustBePositive,
    #[error("'ν' must be >= dimension")]
    NuMustBeGTEDimension,
    #[error("Unknown error")]
    Unknown,
}

impl Distribution for NormalInverseWishart {
    type Value = ExactMultivariateNormalParams;
    type Condition = NormalInverseWishartParams;

    fn fk(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let mu0 = theta.mu0().clone();
        let lambda = theta.lambda();
        let lpsi = theta.lpsi().clone();
        let nu = theta.nu();
        let dim = mu0.len();

        let mu = x.mu();
        let lsigma = x.lsigma();

        let n = MultivariateNormal::new();
        let w_inv = InverseWishart;

        Ok(n.fk(
            mu,
            &ExactMultivariateNormalParams::new(
                mu0,
                PPTRF(
                    SymmetricPackedMatrix::from(
                        dim,
                        ((1.0 / lambda).sqrt() * lsigma.0.elems().to_vec().col_mat()).vec(),
                    )
                    .unwrap(),
                ),
            )?,
        )? * w_inv.fk(lsigma, &InverseWishartParams::new(lpsi, nu)?)?)
    }

    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let mu0 = theta.mu0().clone();
        let lambda = theta.lambda();
        let lpsi = theta.lpsi().clone();
        let nu = theta.nu();
        let dim = mu0.len();

        let n = MultivariateNormal::new();
        let winv = InverseWishart;

        let lsigma = winv.sample(&InverseWishartParams::new(lpsi, nu)?, rng)?;
        let mu = n.sample(
            &ExactMultivariateNormalParams::new(
                mu0,
                PPTRF(
                    SymmetricPackedMatrix::from(
                        dim,
                        ((1.0 / lambda).sqrt() * lsigma.0.elems().to_vec().col_mat()).vec(),
                    )
                    .unwrap(),
                ),
            )?,
            rng,
        )?;

        Ok(ExactMultivariateNormalParams::new(mu, lsigma)?)
    }
}

impl<Rhs, TRhs> Mul<Rhs> for NormalInverseWishart
where
    Rhs: Distribution<Value = TRhs, Condition = NormalInverseWishartParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<
        Self,
        Rhs,
        ExactMultivariateNormalParams,
        TRhs,
        NormalInverseWishartParams,
    >;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for NormalInverseWishart
where
    Rhs: Distribution<Value = NormalInverseWishartParams, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output =
        DependentJoint<Self, Rhs, ExactMultivariateNormalParams, NormalInverseWishartParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
