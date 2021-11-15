pub mod measure;
pub mod pitman_yor_process;
pub mod stick_breaking_process;

pub use measure::*;
pub use pitman_yor_process::*;
pub use stick_breaking_process::*;

use super::BaselineMeasure;
use crate::{Distribution, DistributionError, RandomVariable};
use std::{fmt::Debug, marker::PhantomData};

/// Using stick breaking process.
#[derive(Clone, Debug)]
pub struct DirichletProcess<G0, T>
where
    G0: Distribution<Value = T, Condition = ()>,
    T: RandomVariable,
{
    phantom: PhantomData<(G0, T)>,
}

#[derive(thiserror::Error, Debug)]
pub enum DirichletProcessError {
    #[error("'α' must be positibe")]
    AlphaMustBePositive,
    #[error("Unknown error")]
    Unknown,
}

impl<G0, T> DirichletProcess<G0, T>
where
    G0: Distribution<Value = T, Condition = ()>,
    T: RandomVariable,
{
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

// impl<G0, T> Distribution for DirichletProcess<G0, T>
// where
//     G0: Distribution<T = T, U = ()>,
//     T: RandomVariable,
// {
//     type T = DirichletRandomMeasure<T>;
//     type U = DirichletProcessParams<G0, T>;

//     fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {}

//     fn sample(
//         &self,
//         theta: &Self::U,
//         rng: &mut dyn RngCoreand::prelude::StdRng,
//     ) -> Result<Self::T, DistributionError> {
//     }
// }

#[derive(Clone, Debug)]
pub struct DirichletProcessParams<G0, T>
where
    G0: Distribution<Value = T, Condition = ()>,
    T: RandomVariable,
{
    alpha: f64,
    g0: BaselineMeasure<G0, T>,
}

impl<G0, T> DirichletProcessParams<G0, T>
where
    G0: Distribution<Value = T, Condition = ()>,
    T: RandomVariable,
{
    pub fn new(alpha: f64, g0: BaselineMeasure<G0, T>) -> Result<Self, DistributionError> {
        if alpha <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                DirichletProcessError::AlphaMustBePositive.into(),
            ));
        }

        Ok(Self { alpha, g0 })
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn g0(&self) -> &BaselineMeasure<G0, T> {
        &self.g0
    }
}
