use super::wishart::{Wishart, WishartParams};
use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use opensrdk_linear_algebra::*;
use rand::prelude::*;
use special::Gamma;
use std::f64::consts::PI;
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

fn multivariate_gamma(p: u64, a: f64) -> f64 {
    PI.powf(p as f64 * (p as f64 - 1.0) / 4.0)
        * (0..p)
            .into_iter()
            .map(|i| Gamma::gamma(a + i as f64 / 2.0))
            .product::<f64>()
}

impl Distribution for InverseWishart {
    type T = Matrix;
    type U = InverseWishartParams;

    /// x must be cholesky decomposed
    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let lpsi = theta.lpsi();
        let nu = theta.nu();

        let p = x.rows() as f64;

        Ok(lpsi.trdet().powf(nu / 2.0)
            / (2f64.powf(nu * p / 2.0) * multivariate_gamma(p as u64, nu / 2.0))
            * x.trdet().powf(-(nu + p + 1.0) / 2.0)
            * (-0.5 * x.clone().potrs(lpsi * lpsi.t())?.tr()).exp())
    }

    /// output is cholesky decomposed
    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, DistributionError> {
        let lpsi = theta.lpsi();
        let nu = theta.nu();

        let lpsi_inv = lpsi.clone().potri()?;
        let w = Wishart;
        let w_params = WishartParams::new(lpsi_inv, nu)?;

        let x = w.sample(&w_params, rng)?;
        let x_inv = x.potri()?;

        Ok(x_inv)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InverseWishartParams {
    lpsi: Matrix,
    nu: f64,
}

impl InverseWishartParams {
    pub fn new(lpsi: Matrix, nu: f64) -> Result<Self, DistributionError> {
        let p = lpsi.rows();
        if p != lpsi.cols() {
            return Err(DistributionError::InvalidParameters(
                InverseWishartError::DimensionMismatch.into(),
            ));
        }
        if nu <= p as f64 - 1.0 as f64 {
            return Err(DistributionError::InvalidParameters(
                InverseWishartError::NuMustBeGTEDimension.into(),
            ));
        }

        Ok(Self { lpsi, nu })
    }

    pub fn lpsi(&self) -> &Matrix {
        &self.lpsi
    }

    pub fn nu(&self) -> f64 {
        self.nu
    }
}

impl<Rhs, TRhs> Mul<Rhs> for InverseWishart
where
    Rhs: Distribution<T = TRhs, U = InverseWishartParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, Matrix, TRhs, InverseWishartParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for InverseWishart
where
    Rhs: Distribution<T = InverseWishartParams, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, Matrix, InverseWishartParams, URhs>;

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
