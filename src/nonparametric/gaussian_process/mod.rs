pub mod exact_gp;
pub mod ey;
pub mod kernel_matrix;
pub mod kiss_love_gp;
pub mod regressor;
pub mod student_tp;

use crate::{Distribution, RandomVariable};
pub use exact_gp::*;
pub use ey::*;
pub use kernel_matrix::*;
pub use kiss_love_gp::*;
use opensrdk_kernel_method::Kernel;
pub use regressor::*;
use std::{error::Error, fmt::Debug};
pub use student_tp::*;

#[derive(thiserror::Error, Debug)]
pub enum GaussianProcessError {
    #[error("Data is empty.")]
    Empty,
    #[error("Dimension mismatch.")]
    DimensionMismatch,
    #[error("NaN contaminated.")]
    NaNContamination,
}

///
/// ![tex](https://latex.codecogs.com/svg.latex?y_n%3Df%28\mathbf{x}_n%29+\varepsilon_n)
///
/// ![tex](https://latex.codecogs.com/svg.latex?\mathbf{f}%7CX\sim\mathcal{GP}%280,K_{XX}%29)
///
pub trait GaussianProcess<K, T>: Distribution
where
    K: Kernel<T>,
    T: RandomVariable,
{
    fn new(kernel: K) -> Self;

    fn kxx_inv_vec(
        &self,
        vec: Vec<f64>,
        params: &GaussianProcessParams<T>,
        with_det_lkxx: bool,
    ) -> Result<(Vec<f64>, Option<f64>), Box<dyn Error>>;

    fn lkxx_vec(
        &self,
        vec: Vec<f64>,
        params: &GaussianProcessParams<T>,
    ) -> Result<Vec<f64>, Box<dyn Error>>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct GaussianProcessParams<T>
where
    T: RandomVariable,
{
    x: Vec<T>,
    theta: Vec<f64>,
}

impl<T> GaussianProcessParams<T>
where
    T: RandomVariable,
{
    pub fn new(x: Vec<T>, theta: Vec<f64>) -> Self {
        Self { x, theta }
    }

    pub fn eject(self) -> (Vec<T>, Vec<f64>) {
        (self.x, self.theta)
    }
}
