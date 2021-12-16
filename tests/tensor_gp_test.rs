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

fn fk(x: Vec<T>, z: Vec<Vec<f64>>) -> Result<Vec<f64>, DistributionError> {
    let zi_len = z[0].len();
    let n = x.len();
    let y_zero = vec![0.0; n];
    let kernel = RBF + Periodic;
    let theta = vec![1.0; kernel.params_len()];

    // sigma, lsigmaをconditionを使って受け取る形に変更する
    let sigma = 1.0;
    let lsigma = Matrix::from(zi_len, vec![1.0; zi_len * zi_len])?;
    let distr_zi = MultivariateNormal::new().condition(|yi: &f64| {
        ExactMultivariateNormalParams::new((*yi * vec![1.0; zi_len].col_mat()).vec(), lsigma)
    });
    let distr_z = vec![distr_zi; n].into_iter().joint();
    let params_y = BaseEllipticalProcessParams::new(kernel, x, theta, sigma)?.exact(&y_zero)?;
    let distr_y = MultivariateNormal::new().condition(|_: &()| {
        BaseEllipticalProcessParams::new(kernel, x, theta, sigma)?.exact(&y_zero)
    });
    let distr_zy = distr_z & distr_y;

    let pre_distr_sigma = InstantDistribution::new(
        |x: &f64, _theta: &()| {
            let mu = x.mean();
            if x < 0 {
                let p = 0;
            } else {
                let p = (-(x - mu).powi(2) / (2.0 * sigma.powi(2))).exp() * 2;
            }
            Ok(p)
        },
        |_theta| {
            let mut rng = StdRng::from_seed([1; 32]);
            Normal
                .sample(
                    &NormalParams::new(10.0, (10.0f64 * 0.1).abs()).unwrap(),
                    rng,
                )
                .abs();
        },
    );
    let mu0 = z.iter().mean().collect::<Result<Vec<f64>, _>>().unwrap();
    let lambda = 1.0;
    let lpsi = Matrix::from(zi_len, vec![1.0; zi_len * zi_len])?;
    let nu = zi_len as f64;
    let pre_distr_lsigma = NormalInverseWishart::new(mu0, lambda, lpsi, nu);
    let distr = distr_zy & pre_distr_lsigma & pre_distr_sigma;
    //　で、MCMC使ってy, sigma, lsigmaを求めてやる
}
