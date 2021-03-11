use super::multivariate_normal::{MultivariateNormal, MultivariateNormalParams};
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use opensrdk_linear_algebra::*;
use rand::prelude::*;
use special::Gamma;
use std::f64::consts::PI;
use std::{error::Error, ops::BitAnd, ops::Mul};

/// # Wishart
#[derive(Clone, Debug)]
pub struct Wishart;

#[derive(thiserror::Error, Debug)]
pub enum WishartError {
    #[error("Dimension mismatch")]
    DimensionMismatch,
    #[error("'n' must be >= dimension")]
    NMustBeGTEDimension,
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

impl Distribution for Wishart {
    type T = Matrix;
    type U = WishartParams;

    /// x must be cholesky decomposed
    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, Box<dyn Error>> {
        let lv = theta.lv();
        let n = theta.n();

        let p = x.rows() as f64;

        Ok(
            x.trdet().powf((n + p + 1.0) / 2.0) * (-0.5 * lv.clone().potrs(x * x.t())?.tr()).exp()
                / (2f64.powf(n * p / 2.0)
                    * lv.trdet().powf(n / 2.0)
                    * multivariate_gamma(p as u64, n / 2.0)),
        )
    }

    /// output is cholesky decomposed
    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, Box<dyn Error>> {
        let lv = theta.lv();
        let n = theta.n() as usize;

        let p = lv.rows();

        let normal = MultivariateNormal;
        let normal_params = MultivariateNormalParams::new(vec![0.0; p], lv.clone())?;

        let w = (0..n)
            .into_iter()
            .map(|_| normal.sample(&normal_params, rng))
            .try_fold::<Matrix, _, Result<Matrix, Box<dyn Error>>>(
                Matrix::new(p, p),
                |acc, v: Result<Vec<f64>, Box<dyn Error>>| {
                    let v = v?;
                    Ok(acc + v.clone().row_mat() * v.col_mat())
                },
            )?;

        Ok(w)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WishartParams {
    lv: Matrix,
    n: f64,
}

impl WishartParams {
    pub fn new(lv: Matrix, n: f64) -> Result<Self, Box<dyn Error>> {
        let p = lv.rows();
        if p != lv.cols() {
            return Err(WishartError::DimensionMismatch.into());
        }
        if n <= p as f64 - 1.0 as f64 {
            return Err(WishartError::NMustBeGTEDimension.into());
        }

        Ok(Self { lv, n })
    }

    pub fn lv(&self) -> &Matrix {
        &self.lv
    }

    pub fn n(&self) -> f64 {
        self.n
    }
}

impl<Rhs, TRhs> Mul<Rhs> for Wishart
where
    Rhs: Distribution<T = TRhs, U = WishartParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, Matrix, TRhs, WishartParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for Wishart
where
    Rhs: Distribution<T = WishartParams, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, Matrix, WishartParams, URhs>;

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
