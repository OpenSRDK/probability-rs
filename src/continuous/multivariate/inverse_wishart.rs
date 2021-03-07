use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use opensrdk_linear_algebra::*;
use rand::prelude::*;
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

impl Distribution for InverseWishart {
    type T = Vec<f64>;
    type U = InverseWishartParams;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, Box<dyn Error>> {
        let l_psi = theta.l_psi();
        let nu = theta.nu();

        Ok(todo!())
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, Box<dyn Error>> {
        let l_psi = theta.l_psi();
        let nu = theta.nu();

        Ok(todo!())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InverseWishartParams {
    l_psi: Matrix,
    nu: u64,
}

impl InverseWishartParams {
    pub fn new(l_psi: Matrix, nu: u64) -> Result<Self, Box<dyn Error>> {
        let n = l_psi.rows();
        if n != l_psi.cols() {
            return Err(InverseWishartError::DimensionMismatch.into());
        }
        if nu < n as u64 {
            return Err(InverseWishartError::NuMustBeGTEDimension.into());
        }

        Ok(Self { l_psi, nu })
    }

    pub fn l_psi(&self) -> &Matrix {
        &self.l_psi
    }

    pub fn nu(&self) -> u64 {
        self.nu
    }
}

impl<Rhs, TRhs> Mul<Rhs> for InverseWishart
where
    Rhs: Distribution<T = TRhs, U = InverseWishartParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, Vec<f64>, TRhs, InverseWishartParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for InverseWishart
where
    Rhs: Distribution<T = InverseWishartParams, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, Vec<f64>, InverseWishartParams, URhs>;

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
