#[cfg(test)]
extern crate openblas_src;
pub extern crate opensrdk_linear_algebra;
pub extern crate rand;
extern crate rand_distr;
extern crate thiserror;

pub use crate::multivariate_normal::*;
pub use crate::normal::*;
use rand::prelude::*;
use std::error::Error;

pub mod multivariate_normal;
pub mod normal;

pub trait Distribution {
    fn sample(&self, rng: &mut StdRng) -> Result<f64, Box<dyn Error>>;
}

pub trait MultivariateDistribution {
    fn sample(&self, rng: &mut StdRng) -> Result<Vec<f64>, Box<dyn Error>>;
}
