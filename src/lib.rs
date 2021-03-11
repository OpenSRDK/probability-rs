#[cfg(test)]
extern crate blas_src;
#[cfg(test)]
extern crate lapack_src;
extern crate num_integer;
pub extern crate opensrdk_linear_algebra;
pub extern crate rand;
extern crate rand_distr;
extern crate rayon;
extern crate thiserror;

pub mod continuous;
pub mod discrete;
pub mod distribution;
pub mod mcmc;
pub mod nonparametric;

pub use continuous::*;
pub use discrete::*;
pub use distribution::*;
