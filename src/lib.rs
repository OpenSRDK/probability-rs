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
pub use conditional::*;
pub use instant::*;
use rand::prelude::*;
use std::error::Error;

pub mod conditional;
pub mod instant;
pub mod mcmc;
pub mod multivariate_normal;
pub mod normal;

pub trait Distribution<T> {
    fn p(&self, x: &T) -> Result<f64, Box<dyn Error>>;
    fn sample(&self, rng: &mut StdRng) -> Result<T, Box<dyn Error>>;
}
