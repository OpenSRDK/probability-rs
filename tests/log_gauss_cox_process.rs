extern crate blas_src;
extern crate lapack_src;
extern crate opensrdk_kernel_method;
extern crate opensrdk_linear_algebra;
extern crate opensrdk_probability;
extern crate plotters;
extern crate rayon;

use crate::opensrdk_probability::*;
use opensrdk_kernel_method::*;
use opensrdk_linear_algebra::{mat, Matrix, Vector};
use opensrdk_probability::nonparametric::*;
use plotters::{coord::Shift, prelude::*};
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::time::Instant;

#[derive(Clone, Copy)]
pub enum Type {
    Exact,
    Sparse,
    KissLove,
}

#[test]
fn test_main() {}

fn fk(y: Vec<Vec<f64>>) -> Result<Vec<f64>, DistributionError> {
    let kernel = RBF + Periodic;
    let f = Normal;
}
