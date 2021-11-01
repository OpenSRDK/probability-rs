pub mod cluster_switch;
pub mod gibbs;
pub mod gibbs_sampler;

pub use cluster_switch::*;
pub use gibbs::*;
pub use gibbs_sampler::*;

use crate::nonparametric::*;
use crate::RandomVariable;
use crate::*;

#[derive(thiserror::Error, Debug)]
pub enum PitmanYorProcessError {
    #[error("'d' must be greater than or equal to 0 and less than 1")]
    DMustBeGTE0AndLT1,
    #[error("All elements of `s` must be positive")]
    SMustBePositive,
    #[error("`remove_index` is out of range of `s`.")]
    RemoveIndexOutOfRange,
    #[error("Unknown error")]
    Unknown,
}

#[derive(Clone, Debug)]
pub struct PitmanYorProcessParams<G0, TH>
where
    G0: Distribution<T = TH, U = ()>,
    TH: RandomVariable,
{
    alpha: f64,
    d: f64,
    g0: BaselineMeasure<G0, TH>,
}

impl<G0, TH> PitmanYorProcessParams<G0, TH>
where
    G0: Distribution<T = TH, U = ()>,
    TH: RandomVariable,
{
    /// - `d`: 0 ≦ d < 1. If it is zero, Pitman-Yor process means Chinese restaurant process.
    pub fn new(alpha: f64, d: f64, g0: BaselineMeasure<G0, TH>) -> Result<Self, DistributionError> {
        if alpha <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                DirichletProcessError::AlphaMustBePositive.into(),
            ));
        }
        if d < 0.0 || 1.0 <= d {
            return Err(DistributionError::InvalidParameters(
                PitmanYorProcessError::DMustBeGTE0AndLT1.into(),
            ));
        }

        Ok(Self { alpha, d, g0 })
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn d(&self) -> f64 {
        self.d
    }

    pub fn x_in_cluster<T>(x: &[T], s: &[u32], k: u32) -> Vec<T>
    where
        T: RandomVariable,
    {
        s.iter()
            .enumerate()
            .filter(|&(_, &si)| si == k)
            .map(|(i, _)| x[i].clone())
            .collect::<Vec<_>>()
    }
}
