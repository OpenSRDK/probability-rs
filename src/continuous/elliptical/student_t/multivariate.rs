use crate::{
    DependentJoint, Distribution, ExactEllipticalParams, IndependentJoint, RandomVariable,
};
use crate::{DistributionError, EllipticalParams};
use opensrdk_linear_algebra::*;
use rand::prelude::*;
use rand_distr::StudentT as RandStudentT;
use rayon::prelude::*;
use special::Gamma;
use std::f64::consts::PI;
use std::{ops::BitAnd, ops::Mul};

/// # MultivariateStudentT
#[derive(Clone, Debug)]
pub struct MultivariateStudentT<T = ExactMultivariateStudentTParams, U = ExactEllipticalParams>
where
    T: MultivariateStudentTParams<U>,
    U: EllipticalParams;

#[derive(thiserror::Error, Debug)]
pub enum MultivariateStudentTError {
    #[error("dimension mismatch")]
    DimensionMismatch,
}

impl<T, U> Distribution for MultivariateStudentT<T, U>
where
    T: MultivariateStudentTParams<U>,
    U: EllipticalParams,
{
    type T = Vec<f64>;
    type U = T;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let elliptical = theta.elliptical();
        let x_mu = elliptical.x_mu(x)?;

        let n = x_mu.len() as f64;
        let nu = theta.nu();

        Ok((Gamma::gamma((nu + n) / 2.0)
            / (Gamma::gamma(nu / 2.0)
                * nu.powf(n / 2.0)
                * PI.powf(n / 2.0)
                * elliptical.lsigma_det()))
            * (1.0 + elliptical.x_mu_t_sigma_inv_x_mu(x_mu)? / nu).powf(-(nu + n) / 2.0))
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, DistributionError> {
        let nu = theta.nu();
        let elliptical = theta.elliptical();

        let student_t = match RandStudentT::new(nu) {
            Ok(v) => Ok(v),
            Err(e) => Err(DistributionError::Others(e.into())),
        }?;

        let z = (0..elliptical.z_len_for_sample())
            .into_iter()
            .map(|_| rng.sample(student_t))
            .collect::<Vec<_>>();

        let y = elliptical.sample_from_z(&z)?;

        Ok(y.vec())
    }
}

pub trait MultivariateStudentTParams<T>
where
    T: EllipticalParams,
{
    fn nu(&self) -> f64;
    fn elliptical(&self) -> &T;
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactMultivariateStudentTParams {
    nu: f64,
    elliptical: ExactEllipticalParams,
}

impl ExactMultivariateStudentTParams {
    /// # Multivariate student t
    /// `L` is needed as second argument under decomposition `Sigma = L * L^T`
    /// lsigma = sigma.potrf()?;
    pub fn new(nu: f64, mu: Vec<f64>, lsigma: Matrix) -> Result<Self, DistributionError> {
        let elliptical = ExactEllipticalParams::new(mu, lsigma)?;

        Ok(Self { nu, elliptical })
    }

    pub fn nu(&self) -> f64 {
        self.nu
    }

    pub fn mu(&self) -> &Vec<f64> {
        self.elliptical.mu()
    }

    pub fn lsigma(&self) -> &Matrix {
        self.elliptical.lsigma()
    }
}

impl<T> MultivariateStudentTParams<T> for ExactMultivariateStudentTParams
where
    T: EllipticalParams,
{
    fn nu(&self) -> f64 {
        self.nu
    }

    fn elliptical(&self) -> &T {
        &self.elliptical
    }
}

impl<T, U, Rhs, TRhs> Mul<Rhs> for MultivariateStudentT<T, U>
where
    T: MultivariateStudentTParams<U>,
    U: EllipticalParams,
    Rhs: Distribution<T = TRhs, U = T>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, Vec<f64>, TRhs, T>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<T, U, Rhs, URhs> BitAnd<Rhs> for MultivariateStudentT<T, U>
where
    T: MultivariateStudentTParams<U>,
    U: EllipticalParams,
    Rhs: Distribution<T = T, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, Vec<f64>, T, URhs>;

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
