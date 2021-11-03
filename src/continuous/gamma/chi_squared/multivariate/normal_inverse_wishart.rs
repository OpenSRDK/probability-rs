use crate::DistributionError;
use crate::{
    DependentJoint, Distribution, ExactMultivariateNormalParams, IndependentJoint, InverseWishart,
    InverseWishartParams, MultivariateNormal, RandomVariable,
};
use opensrdk_linear_algebra::*;
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
    type T = ExactMultivariateNormalParams;
    type U = NormalInverseWishartParams;

    fn fk(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let mu0 = theta.mu0().clone();
        let lambda = theta.lambda();
        let lpsi = theta.lpsi().clone();
        let nu = theta.nu();

        let mu = x.mu();
        let lsigma = x.lsigma();

        let n = MultivariateNormal::new();
        let w_inv = InverseWishart;

        Ok(n.fk(
            mu,
            &ExactMultivariateNormalParams::new(mu0, (1.0 / lambda) * lsigma.clone())?,
        )? * w_inv.fk(lsigma, &InverseWishartParams::new(lpsi, nu)?)?)
    }

    fn sample(&self, theta: &Self::U, rng: &mut dyn RngCore) -> Result<Self::T, DistributionError> {
        let mu0 = theta.mu0().clone();
        let lambda = theta.lambda();
        let lpsi = theta.lpsi().clone();
        let nu = theta.nu();

        let p = MultivariateNormal::new();
        let winv = InverseWishart;

        let lsigma = winv.sample(&InverseWishartParams::new(lpsi, nu)?, rng)?;
        let mu = p.sample(
            &ExactMultivariateNormalParams::new(mu0, (1.0 / lambda).sqrt() * lsigma.clone())?,
            rng,
        )?;

        Ok(ExactMultivariateNormalParams::new(mu, lsigma)?)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NormalInverseWishartParams {
    mu0: Vec<f64>,
    lambda: f64,
    lpsi: Matrix,
    nu: f64,
}

impl NormalInverseWishartParams {
    pub fn new(
        mu0: Vec<f64>,
        lambda: f64,
        lpsi: Matrix,
        nu: f64,
    ) -> Result<Self, NormalInverseWishartError> {
        let n = mu0.len();
        if n != lpsi.rows() || n != lpsi.cols() {
            return Err(NormalInverseWishartError::DimensionMismatch.into());
        }
        if lambda <= 0.0 {
            return Err(NormalInverseWishartError::DimensionMismatch.into());
        }
        if nu <= n as f64 - 1.0 {
            return Err(NormalInverseWishartError::NuMustBeGTEDimension.into());
        }

        Ok(Self {
            mu0,
            lambda,
            lpsi,
            nu,
        })
    }

    pub fn mu0(&self) -> &Vec<f64> {
        &self.mu0
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    pub fn lpsi(&self) -> &Matrix {
        &self.lpsi
    }

    pub fn nu(&self) -> f64 {
        self.nu
    }
}

impl<Rhs, TRhs> Mul<Rhs> for NormalInverseWishart
where
    Rhs: Distribution<T = TRhs, U = NormalInverseWishartParams>,
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
    Rhs: Distribution<T = NormalInverseWishartParams, U = URhs>,
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
