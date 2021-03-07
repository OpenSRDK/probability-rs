use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use opensrdk_linear_algebra::*;
use rand::prelude::*;
use std::{error::Error, ops::BitAnd, ops::Mul};

/// # NormalInverseWishart
/// ![tex](https://latex.codecogs.com/svg.latex?\mathcal%7BN%7D%28\mu%2C%20\Sigma%29)
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
    type T = Vec<f64>;
    type U = NormalInverseWishartParams;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, Box<dyn Error>> {
        let mu = theta.mu();
        let lambda = theta.lambda();
        let l_psi = theta.l_psi();
        let nu = theta.nu();

        Ok(todo!())
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, Box<dyn Error>> {
        let mu = theta.mu();
        let lambda = theta.lambda();
        let l_psi = theta.l_psi();
        let nu = theta.nu();

        Ok(todo!())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NormalInverseWishartParams {
    mu: Vec<f64>,
    lambda: f64,
    l_psi: Matrix,
    nu: u64,
}

impl NormalInverseWishartParams {
    pub fn new(mu: Vec<f64>, lambda: f64, l_psi: Matrix, nu: u64) -> Result<Self, Box<dyn Error>> {
        let n = mu.len();
        if n != l_psi.rows() || n != l_psi.cols() {
            return Err(NormalInverseWishartError::DimensionMismatch.into());
        }
        if lambda <= 0.0 {
            return Err(NormalInverseWishartError::DimensionMismatch.into());
        }
        if nu < n as u64 {
            return Err(NormalInverseWishartError::NuMustBeGTEDimension.into());
        }

        Ok(Self {
            mu,
            lambda,
            l_psi,
            nu,
        })
    }

    pub fn mu(&self) -> &Vec<f64> {
        &self.mu
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    pub fn l_psi(&self) -> &Matrix {
        &self.l_psi
    }

    pub fn nu(&self) -> u64 {
        self.nu
    }
}

impl<Rhs, TRhs> Mul<Rhs> for NormalInverseWishart
where
    Rhs: Distribution<T = TRhs, U = NormalInverseWishartParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, Vec<f64>, TRhs, NormalInverseWishartParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for NormalInverseWishart
where
    Rhs: Distribution<T = NormalInverseWishartParams, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, Vec<f64>, NormalInverseWishartParams, URhs>;

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
