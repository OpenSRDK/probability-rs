use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use opensrdk_linear_algebra::*;
use rand::prelude::*;
use rand_distr::StudentT as RandStudentT;
use rayon::prelude::*;
use std::{error::Error, ops::BitAnd, ops::Mul};

/// # MultivariateStudentT
/// ![tex](https://latex.codecogs.com/svg.latex?\mathcal%7BN%7D%28\mu%2C%20\Sigma%29)
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
        let l_sigma = theta.l_sigma();
        let nu = theta.nu();

        let n = x.len();

        if n != mu.len() {
            return Err(MultivariateStudentTError::DimensionMismatch.into());
        }
        let nu = nu;

        let x_mu = x
            .par_iter()
            .zip(mu.par_iter())
            .map(|(&xi, &mui)| xi - mui)
            .collect::<Vec<_>>()
            .col_mat();

        Ok(todo!())
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, Box<dyn Error>> {
        let mu = theta.mu();
        let l_sigma = theta.l_sigma();
        let nu = theta.nu();

        let student_t = RandStudentT::new(nu as f64)?;

        let z = (0..l_sigma.rows())
            .into_iter()
            .map(|_| rng.sample(student_t))
            .collect::<Vec<_>>();

        let y = mu.clone().col_mat().gemm(l_sigma, &z.col_mat(), 1.0, 1.0)?;

        Ok(y.vec())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultivariateStudentTParams {
    mu: Vec<f64>,
    l_sigma: Matrix,
    nu: f64,
}

impl MultivariateStudentTParams {
    /// # Multivariate student t
    /// `L` is needed as second argument under decomposition `Sigma = L * L^T`
    /// l_sigma = sigma.potrf()?;
    pub fn new(mu: Vec<f64>, l_sigma: Matrix, nu: f64) -> Result<Self, Box<dyn Error>> {
        let n = mu.len();
        if n != l_sigma.rows() || n != l_sigma.cols() {
            return Err(MultivariateStudentTError::DimensionMismatch.into());
        }

        Ok(Self { mu, l_sigma, nu })
    }

    pub fn mu(&self) -> &Vec<f64> {
        &self.mu
    }

    pub fn l_sigma(&self) -> &Matrix {
        &self.l_sigma
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
