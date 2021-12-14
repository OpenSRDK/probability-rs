extern crate blas_src;
extern crate lapack_src;
extern crate opensrdk_kernel_method;
extern crate opensrdk_linear_algebra;
extern crate opensrdk_probability;
extern crate plotters;
extern crate rayon;

use crate::opensrdk_probability::*;
use opensrdk_kernel_method::*;
use opensrdk_linear_algebra::mat;
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

fn model() {
    let mu = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let lsigma = mat!(
       1.0,  0.0,  0.0,  0.0,  0.0,  0.0;
       2.0,  3.0,  0.0,  0.0,  0.0,  0.0;
       4.0,  5.0,  6.0,  0.0,  0.0,  0.0;
       7.0,  8.0,  9.0, 10.0,  0.0,  0.0;
      11.0, 12.0, 13.0, 14.0, 15.0,  0.0;
      16.0, 17.0, 18.0, 19.0, 20.0, 21.0
    );

    let model = vec![Normal; 3].into_iter().joint();
    let mut rng = StdRng::from_seed([1; 32]);
}

fn fk(x: Vec<f64>) -> Vec<f64> {
    let params_z = ExactMultivariateNormalParams::new(y * one_vec, lsigma);
    let params_y = ExactMultivariateNormalParams::new(0, karnels);
    let distr_z = MultivariateNormal::new(params_z);
    let distr_y = MultivariateNormal::new(params_y);
    let distr_zy = independent_array_joint(distr_y, distr_z);
    let pre_distr_sigma = InstantDistribution::new(p, sample);
    let pre_distr_lsigma = NormalInverseWishart::new(mu0, lambda, lpsi, nu);
    let distr = distr_zy * pre_distr_lsigma * pre_distr_sigma;
    //　で、MCMC使ってy, sigma, lsigmaを求めてやる
}
