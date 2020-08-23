#[cfg(test)]
extern crate openblas_src;
pub extern crate opensrdk_linear_algebra;
pub extern crate rand;
extern crate rand_distr;

pub mod multivariate_normal;
pub mod normal;

pub use crate::multivariate_normal::*;
pub use crate::normal::*;
use rand::prelude::*;

pub trait Distribution {
    fn sample(&self, rng: &mut StdRng) -> Result<f64, String>;
}

pub trait MultivariateDistribution {
    fn sample(&self, rng: &mut StdRng) -> Result<Vec<f64>, String>;
}
