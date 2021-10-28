pub mod chinese_restaurant_process;
pub mod pitman_yor_process;

use std::marker::PhantomData;

pub use chinese_restaurant_process::*;
pub use pitman_yor_process::*;

use super::{BaselineMeasure, DiscreteMeasurableSpace, DiscreteMeasure};
use crate::{Distribution, DistributionError, RandomVariable};

#[derive(Clone, Debug)]
pub struct DirichletProcess<D, T, U, G0, TH>
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
    G0: Distribution<T = TH, U = ()>,
    TH: RandomVariable,
{
    distr: D,
    phantom: PhantomData<(G0, TH)>,
}

#[derive(thiserror::Error, Debug)]
pub enum DirichletProcessError {
    #[error("'α' must be positibe")]
    AlphaMustBePositive,
    #[error("Unknown error")]
    Unknown,
}

impl<D, T, U, G0, TH> DirichletProcess<D, T, U, G0, TH>
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
    G0: Distribution<T = TH, U = ()>,
    TH: RandomVariable,
{
    pub fn new(distr: D) -> Self {
        Self {
            distr,
            phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DirichletRandomMeasure<T>
where
    T: RandomVariable,
{
    pi_theta: Vec<(f64, T)>,
}

impl<T> DiscreteMeasure for DirichletRandomMeasure<T>
where
    T: RandomVariable,
{
    fn measure(&self, a: DiscreteMeasurableSpace) -> f64 {
        a.iter().map(|(&i, ())| self.pi_theta[i].0).sum()
    }
}

impl<T> DirichletRandomMeasure<T>
where
    T: RandomVariable,
{
    pub fn new(pi_theta: Vec<(f64, T)>) -> Self {
        Self { pi_theta }
    }

    pub fn pi_theta(&self) -> &Vec<(f64, T)> {
        &self.pi_theta
    }
}

impl<D, T, U, G0, TH> Distribution for DirichletProcess<D, T, U, G0, TH>
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
    G0: Distribution<T = TH, U = ()>,
    TH: RandomVariable,
{
    type T = DirichletRandomMeasure<TH>;
    type U = DirichletProcessParams<G0, TH>;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        todo!("{:?}{:?}", x, theta)
    }

    fn sample(
        &self,
        theta: &Self::U,
        rng: &mut rand::prelude::StdRng,
    ) -> Result<Self::T, DistributionError> {
        self.distr.sample(theta, rng)
    }
}

#[derive(Clone, Debug)]
pub struct DirichletProcessParams<D, T>
where
    D: Distribution<T = T, U = ()>,
    T: RandomVariable,
{
    alpha: f64,
    g0: BaselineMeasure<D, T>,
}

impl<D, T> DirichletProcessParams<D, T>
where
    D: Distribution<T = T, U = ()>,
    T: RandomVariable,
{
    pub fn new(alpha: f64, g0: BaselineMeasure<D, T>) -> Result<Self, DistributionError> {
        if alpha <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                DirichletProcessError::AlphaMustBePositive.into(),
            ));
        }

        Ok(Self { alpha, g0 })
    }
}
