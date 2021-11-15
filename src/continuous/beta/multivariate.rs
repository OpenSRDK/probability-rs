use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use rand_distr::Dirichlet as RandDirichlet;
use rayon::{iter::IntoParallelIterator, prelude::*};
use std::{ops::BitAnd, ops::Mul};

/// Dirichlet distribution
#[derive(Clone, Debug)]
pub struct Dirichlet;

#[derive(thiserror::Error, Debug)]
pub enum DirichletError {
    #[error("Dimension mismatch")]
    DimensionMismatch,
    #[error("Length of 'α' must be >= 2")]
    AlphaLenMustBeGTE2,
    #[error("'α' must be positibe")]
    AlphaMustBePositive,
    #[error("Unknown error")]
    Unknown,
}

impl Distribution for Dirichlet {
    type Value = Vec<f64>;
    type Condition = DirichletParams;

    fn fk(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let alpha = theta.alpha();

        if x.len() != alpha.len() {
            return Err(DistributionError::InvalidParameters(
                DirichletError::DimensionMismatch.into(),
            ));
        }

        Ok(x.into_par_iter()
            .zip(alpha.into_par_iter())
            .map(|(&xi, &alphai)| xi.powf(alphai - 1.0))
            .product::<f64>())
    }

    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let alpha = theta.alpha();

        let dirichlet = match RandDirichlet::new(alpha) {
            Ok(n) => n,
            Err(e) => return Err(DistributionError::Others(e.into())),
        };

        Ok(rng.sample(dirichlet))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DirichletParams {
    alpha: Vec<f64>,
}

impl DirichletParams {
    pub fn new(alpha: Vec<f64>) -> Result<Self, DistributionError> {
        if alpha.len() < 2 {
            return Err(DistributionError::InvalidParameters(
                DirichletError::AlphaLenMustBeGTE2.into(),
            ));
        }
        for &alpha_i in alpha.iter() {
            if alpha_i <= 0.0 {
                return Err(DistributionError::InvalidParameters(
                    DirichletError::AlphaMustBePositive.into(),
                ));
            }
        }

        Ok(Self { alpha })
    }

    pub fn alpha(&self) -> &[f64] {
        &self.alpha
    }
}

impl<Rhs, TRhs> Mul<Rhs> for Dirichlet
where
    Rhs: Distribution<Value = TRhs, Condition = DirichletParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, Vec<f64>, TRhs, DirichletParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for Dirichlet
where
    Rhs: Distribution<Value = DirichletParams, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, Vec<f64>, DirichletParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Dirichlet, DirichletParams, Distribution};
    use rand::prelude::*;
    #[test]
    fn it_works() {
        let dirichlet = Dirichlet;
        let mut rng = StdRng::from_seed([1; 32]);

        let alpha = vec![1.0, 1.0, 2.0, 3.0, 4.0, 5.0];

        let x = dirichlet
            .sample(&DirichletParams::new(alpha).unwrap(), &mut rng)
            .unwrap();

        println!("{:#?}", x);
    }
}
