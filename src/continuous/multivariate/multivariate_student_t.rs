use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use opensrdk_linear_algebra::*;
use rand::prelude::*;
use rand_distr::StudentT as RandStudentT;
use rayon::prelude::*;
use special::Gamma;
use std::f64::consts::PI;
use std::{error::Error, ops::BitAnd, ops::Mul};

/// # MultivariateStudentT
#[derive(Clone, Debug)]
pub struct MultivariateStudentT;

#[derive(thiserror::Error, Debug)]
pub enum MultivariateStudentTError {
    #[error("dimension mismatch")]
    DimensionMismatch,
}

impl Distribution for MultivariateStudentT {
    type T = Vec<f64>;
    type U = MultivariateStudentTParams;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, Box<dyn Error>> {
        let mu = theta.mu();
        let lsigma = theta.lsigma();
        let nu = theta.nu();

        let p = x.len();

        if p != mu.len() {
            return Err(MultivariateStudentTError::DimensionMismatch.into());
        }
        let p = p as f64;
        let nu = nu;

        let x_mu = x
            .par_iter()
            .zip(mu.par_iter())
            .map(|(&xi, &mui)| xi - mui)
            .collect::<Vec<_>>()
            .col_mat();

        Ok((Gamma::gamma((nu + p) / 2.0)
            / (Gamma::gamma(nu / 2.0) * nu.powf(p / 2.0) * PI.powf(p / 2.0) * lsigma.trdet()))
            * (1.0 + (x_mu.t() * lsigma.potrs(x_mu)?)[0][0] / nu).powf(-(nu + p) / 2.0))
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, Box<dyn Error>> {
        let mu = theta.mu();
        let lsigma = theta.lsigma();
        let nu = theta.nu();

        let student_t = RandStudentT::new(nu as f64)?;

        let z = (0..lsigma.rows())
            .into_iter()
            .map(|_| rng.sample(student_t))
            .collect::<Vec<_>>();

        let y = mu.clone().col_mat().gemm(lsigma, &z.col_mat(), 1.0, 1.0)?;

        Ok(y.vec())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultivariateStudentTParams {
    mu: Vec<f64>,
    lsigma: Matrix,
    nu: f64,
}

impl MultivariateStudentTParams {
    /// # Multivariate student t
    /// `L` is needed as second argument under decomposition `Sigma = L * L^T`
    /// lsigma = sigma.potrf()?;
    pub fn new(mu: Vec<f64>, lsigma: Matrix, nu: f64) -> Result<Self, Box<dyn Error>> {
        let n = mu.len();
        if n != lsigma.rows() || n != lsigma.cols() {
            return Err(MultivariateStudentTError::DimensionMismatch.into());
        }

        Ok(Self { mu, lsigma, nu })
    }

    pub fn mu(&self) -> &Vec<f64> {
        &self.mu
    }

    pub fn lsigma(&self) -> &Matrix {
        &self.lsigma
    }

    pub fn nu(&self) -> f64 {
        self.nu
    }
}

impl<Rhs, TRhs> Mul<Rhs> for MultivariateStudentT
where
    Rhs: Distribution<T = TRhs, U = MultivariateStudentTParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, Vec<f64>, TRhs, MultivariateStudentTParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for MultivariateStudentT
where
    Rhs: Distribution<T = MultivariateStudentTParams, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, Vec<f64>, MultivariateStudentTParams, URhs>;

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
