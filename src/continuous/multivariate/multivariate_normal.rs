use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use opensrdk_linear_algebra::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::{error::Error, f64::consts::PI, ops::BitAnd, ops::Mul};

/// # MultivariateNormal
/// ![tex](https://latex.codecogs.com/svg.latex?\mathcal%7BN%7D%28\mu%2C%20\Sigma%29)
#[derive(Clone, Debug)]
pub struct MultivariateNormal;

#[derive(thiserror::Error, Debug)]
pub enum MultivariateNormalError {
    #[error("dimension mismatch")]
    DimensionMismatch,
}

impl Distribution for MultivariateNormal {
    type T = Vec<f64>;
    type U = MultivariateNormalParams;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, Box<dyn Error>> {
        let mu = theta.mu();
        let lsigma = theta.lsigma();

        let n = x.len();

        if n != mu.len() {
            return Err(MultivariateNormalError::DimensionMismatch.into());
        }
        let n = n as f64;

        let x_mu = x
            .par_iter()
            .zip(mu.par_iter())
            .map(|(&xi, &mui)| xi - mui)
            .collect::<Vec<_>>()
            .col_mat();

        Ok(1.0 / ((2.0 * PI).powf(n / 2.0) * lsigma.trdet())
            * (-1.0 / 2.0 * (x_mu.t() * lsigma.potrs(x_mu)?)[0][0]).exp())
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, Box<dyn Error>> {
        let mu = theta.mu();
        let lsigma = theta.lsigma();

        let z = (0..lsigma.rows())
            .into_iter()
            .map(|_| rng.sample(StandardNormal))
            .collect::<Vec<_>>();

        let y = mu.clone().col_mat().gemm(lsigma, &z.col_mat(), 1.0, 1.0)?;

        Ok(y.vec())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultivariateNormalParams {
    mu: Vec<f64>,
    lsigma: Matrix,
}

impl MultivariateNormalParams {
    /// # Multivariate normal
    /// `L` is needed as second argument under decomposition `Sigma = L * L^T`
    /// l_sigma = sigma.potrf()?;
    pub fn new(mu: Vec<f64>, lsigma: Matrix) -> Result<Self, Box<dyn Error>> {
        let n = mu.len();
        if n != lsigma.rows() || n != lsigma.cols() {
            return Err(MultivariateNormalError::DimensionMismatch.into());
        }

        Ok(Self { mu, lsigma })
    }

    pub fn mu(&self) -> &Vec<f64> {
        &self.mu
    }

    pub fn lsigma(&self) -> &Matrix {
        &self.lsigma
    }

    pub fn eject(self) -> (Vec<f64>, Matrix) {
        (self.mu, self.lsigma)
    }
}

impl<Rhs, TRhs> Mul<Rhs> for MultivariateNormal
where
    Rhs: Distribution<T = TRhs, U = MultivariateNormalParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, Vec<f64>, TRhs, MultivariateNormalParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for MultivariateNormal
where
    Rhs: Distribution<T = MultivariateNormalParams, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, Vec<f64>, MultivariateNormalParams, URhs>;

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
