use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use opensrdk_linear_algebra::*;
use rand::prelude::*;
use special::Gamma;
use std::f64::consts::PI;
use std::{error::Error, ops::BitAnd, ops::Mul};

/// # InverseWishart
/// ![tex](https://latex.codecogs.com/svg.latex?\mathcal%7BN%7D%28\mu%2C%20\Sigma%29)
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

fn multivariate_gamma(n: u64, a: f64) -> f64 {
    PI.powf(n as f64 * (n as f64 - 1.0) / 4.0)
        * (0..n)
            .into_iter()
            .map(|i| Gamma::gamma(a + i as f64 / 2.0))
            .product::<f64>()
}

impl Distribution for InverseWishart {
    type T = Matrix;
    type U = InverseWishartParams;

    /// x must be cholesky decomposed
    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, Box<dyn Error>> {
        let lpsi = theta.lpsi();
        let nu = theta.nu();

        let n = x.rows() as f64;

        Ok(lpsi.trdet().powf(nu / 2.0)
            / (2f64.powf(nu * n / 2.0) * multivariate_gamma(n as u64, nu / 2.0))
            * x.trdet().powf(-(nu + n + 1.0) / 2.0)
            * (-0.5 * x.clone().potrs(lpsi * lpsi.t())?.tr()).exp())
    }

    /// output is cholesky decomposed
    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, Box<dyn Error>> {
        let lpsi = theta.lpsi();
        let nu = theta.nu();

        Ok(todo!())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InverseWishartParams {
    lpsi: Matrix,
    nu: f64,
}

impl InverseWishartParams {
    pub fn new(lpsi: Matrix, nu: f64) -> Result<Self, Box<dyn Error>> {
        let n = lpsi.rows();
        if n != lpsi.cols() {
            return Err(InverseWishartError::DimensionMismatch.into());
        }
        if nu <= n as f64 - 1.0 as f64 {
            return Err(InverseWishartError::NuMustBeGTEDimension.into());
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
