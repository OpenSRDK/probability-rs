use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable, VectorSampleable};
use crate::{DistributionError, StudentTError};
use rand::prelude::*;
use rand_distr::StudentT as RandStudentT;
use special::Gamma;
use std::f64::consts::PI;
use std::{ops::BitAnd, ops::Mul};

/// Student-t distribution
#[derive(Clone, Debug)]
pub struct StudentT;

impl Distribution for StudentT {
    type T = f64;
    type U = StudentTParams;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let nu = theta.nu();
        let mu = theta.mu();
        let sigma = theta.sigma();

        Ok(
            (Gamma::gamma((nu + 1.0) / 2.0) / ((nu * PI).sqrt() * Gamma::gamma(nu / 2.0)))
                * (1.0 + ((x - mu) / sigma).powi(2) / nu).powf(-((nu + 1.0) / 2.0)),
        )
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, DistributionError> {
        let nu = theta.nu();

        let student_t = match RandStudentT::new(nu) {
            Ok(v) => Ok(v),
            Err(e) => Err(DistributionError::Others(e.into())),
        }?;

        Ok(rng.sample(student_t))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StudentTParams {
    nu: f64,
    mu: f64,
    sigma: f64,
}

impl StudentTParams {
    pub fn new(nu: f64, mu: f64, sigma: f64) -> Result<Self, DistributionError> {
        if sigma <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                StudentTError::SigmaMustBePositive.into(),
            ));
        }
        Ok(Self { nu, mu, sigma })
    }

    pub fn nu(&self) -> f64 {
        self.nu
    }

    pub fn mu(&self) -> f64 {
        self.mu
    }

    pub fn sigma(&self) -> f64 {
        self.sigma
    }
}

impl<Rhs, TRhs> Mul<Rhs> for StudentT
where
    Rhs: Distribution<T = TRhs, U = StudentTParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, f64, TRhs, StudentTParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for StudentT
where
    Rhs: Distribution<T = StudentTParams, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, f64, StudentTParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

impl VectorSampleable for StudentTParams {
    type T = ();
    fn transform_vec(self) -> (Vec<f64>, Self::T) {
        (vec![self.nu, self.mu, self.sigma], ())
    }
    fn restore(v: (Vec<f64>, Self::T)) -> Self {
        Self::new(v.0[0], v.0[1], v.0[2]).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::{Distribution, StudentT, StudentTParams};
    use rand::prelude::*;

    #[test]
    fn it_works() {
        let n = StudentT;
        let mut rng = StdRng::from_seed([1; 32]);

        let mu = 2.0;
        let sigma = 3.0;

        let x = n
            .sample(&StudentTParams::new(1.0, mu, sigma).unwrap(), &mut rng)
            .unwrap();

        println!("{}", x);
    }
}
