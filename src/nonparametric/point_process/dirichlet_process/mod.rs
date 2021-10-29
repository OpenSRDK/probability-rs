pub mod pitman_yor_gibbs;
pub mod pitman_yor_process;

pub use pitman_yor_gibbs::*;
pub use pitman_yor_process::*;

use super::{BaselineMeasure, DiscreteMeasurableSpace, DiscreteMeasure};
use crate::{Distribution, DistributionError, RandomVariable};
use std::fmt::Debug;

pub trait DirichletProcess<T, G0, TH>:
    Clone + Debug + Distribution<T = DirichletRandomMeasure<TH>, U = T>
where
    T: DirichletProcessParams<G0, TH>,
    G0: Distribution<T = TH, U = ()>,
    TH: RandomVariable,
{
    fn z_compaction(z: &mut Vec<usize>, n_vec: &Vec<usize>) {
        for (j, &nj) in n_vec.iter().enumerate() {
            if nj != 0 {
                continue;
            }

            for zi in z.iter_mut().filter(|zi| **zi >= j) {
                *zi -= 1;
            }
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum DirichletProcessError {
    #[error("'α' must be positibe")]
    AlphaMustBePositive,
    #[error("Unknown error")]
    Unknown,
}

#[derive(Clone, Debug)]
pub struct DirichletRandomMeasure<T>
where
    T: RandomVariable,
{
    w_theta: Vec<(usize, T)>,
    z: Vec<usize>,
}

impl<T> DiscreteMeasure for DirichletRandomMeasure<T>
where
    T: RandomVariable,
{
    fn measure(&self, a: DiscreteMeasurableSpace) -> f64 {
        a.iter()
            .map(|(&i, ())| self.w_theta[i].0 as f64)
            .sum::<f64>()
            / self.z.len() as f64
    }
}

impl<T> DirichletRandomMeasure<T>
where
    T: RandomVariable,
{
    pub fn new(w_theta: Vec<(usize, T)>, z: Vec<usize>) -> Self {
        Self { w_theta, z }
    }

    pub fn w_theta(&self) -> &Vec<(usize, T)> {
        &self.w_theta
    }

    pub fn z(&self) -> &Vec<usize> {
        &self.z
    }
}

#[derive(Clone, Debug)]
pub struct BaseDirichletProcessParams<G0, T>
where
    G0: Distribution<T = T, U = ()>,
    T: RandomVariable,
{
    alpha: f64,
    g0: BaselineMeasure<G0, T>,
}

impl<G0, T> BaseDirichletProcessParams<G0, T>
where
    G0: Distribution<T = T, U = ()>,
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
}

impl<G0, T> DirichletProcessParams<G0, T> for BaseDirichletProcessParams<G0, T>
where
    G0: Distribution<T = T, U = ()>,
    T: RandomVariable,
{
    fn alpha(&self) -> f64 {
        self.alpha
    }

    fn g0(&self) -> &BaselineMeasure<G0, T> {
        &self.g0
    }
}

pub trait DirichletProcessParams<G0, T>: RandomVariable
where
    G0: Distribution<T = T, U = ()>,
    T: RandomVariable,
{
    fn alpha(&self) -> f64;
    fn g0(&self) -> &BaselineMeasure<G0, T>;
}
