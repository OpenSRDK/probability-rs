#[cfg(test)]
extern crate blas_src;
#[cfg(test)]
extern crate lapack_src;
pub extern crate opensrdk_linear_algebra;
pub extern crate rand;
extern crate rand_distr;
extern crate rayon;
extern crate thiserror;

pub use crate::multivariate_normal::*;
pub use crate::normal::*;
pub use instant::*;
pub use instant_conditional::*;
use rand::prelude::*;
use std::error::Error;

pub mod instant;
pub mod instant_conditional;
pub mod mcmc;
pub mod multivariate_normal;
pub mod normal;

pub trait Distribution<T> {
    fn p(&self, x: &T) -> Result<f64, Box<dyn Error>>;
    fn sample(&self, rng: &mut StdRng) -> Result<T, Box<dyn Error>>;
}

pub trait ConditionalDistribution<T, U>: Distribution<T> {
    fn with_condition(&mut self, condition: U) -> Result<&mut Self, Box<dyn Error>>;
}
