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

fn fk(x: Vec<Vec<f64>>, z: Matrix) -> Result<Vec<f64>, DistributionError> {
    let zi_len = z.cols();
    let n = x.len();
    let y_zero = vec![0.0; n];
    let kernel = RBF + Periodic;
    let theta = vec![1.0; kernel.params_len()];

    // let lsigma = Matrix::from(zi_len, vec![1.0; zi_len * zi_len])?;
    // let distr_zi = MultivariateNormal::new().condition(|yi: &f64| {
    //     ExactMultivariateNormalParams::new((*yi * vec![1.0; zi_len].col_mat()).vec(), lsigma)
    // });
    let distr_zi = MultivariateNormal::new().condition(&|(yi, lsigma): &(f64, Matrix)| {
        ExactMultivariateNormalParams::new((*yi * vec![1.0; zi_len].col_mat()).vec(), *lsigma)
    });
    let distr_z = vec![distr_zi; n]
        .into_iter()
        .joint()
        .condition(&|(y, lsigma): &(Vec<f64>, Matrix)| {
            Ok(y.iter().map(|yi| (*yi, lsigma.clone())).collect::<Vec<_>>())
        });
    let distr_y = MultivariateNormal::new().condition(&|(sigma, lsigma): &(f64, Matrix)| {
        BaseEllipticalProcessParams::new(kernel, x, theta, *sigma)?.exact(&y_zero)
    });
    // expected struct `Vec<f64>` found struct `Vec<(f64, Matrix)>`
    let distr_zy = distr_z & distr_y;

    let mut rng = StdRng::from_seed([1; 32]);
    let prior_distr_sigma = InstantDistribution::new(
        |x: &f64, _theta: &()| {
            let p = if *x < 0.0 {
                0.0
            } else {
                Normal.fk(x, &NormalParams::new(1.0, 2.0)?)?
            };
            Ok(p)
        },
        |_theta, rng| {
            Ok(Normal
                .sample(
                    &NormalParams::new(10.0, (10.0f64 * 0.1).abs()).unwrap(),
                    &mut rng,
                )?
                .abs())
        },
    );

    let mu0 = mean_fn(z);
    let lambda = 1.0;
    let lpsi = Matrix::from(zi_len, vec![1.0; zi_len * zi_len])?;
    let nu = zi_len as f64;
    let prior_distr_lsigma = NormalInverseWishart
        .condition(|_: &()| NormalInverseWishartParams::new(mu0, lambda, lpsi, nu));
    let prior_distr = prior_distr_lsigma * prior_distr_sigma;
    let sigma0 = 1.0;
    let lsigma0 = Matrix::from(zi_len, vec![1.0; zi_len * zi_len])?;
    // expected struct `Vec<(f64, Matrix)>` found struct `Vec<f64>`

    let sampler = EllipticalSliceSampler::new((sigma0, lsigma0), &distr_zy, &prior_distr);
    Ok(mu0)
}

fn mean_fn(z: Matrix) -> Vec<f64> {
    // let zt = z.t();
    // let sum = Matrix::new(1, zt.rows());
    // for i in 0..zt.cols() {
    //     sum += zt[i];
    // }
    // sum / zt.cols();

    let zt = z.t();
    let z_len = zt.rows();
    (0..z_len)
        .into_iter()
        .map(|i| -> Result<_, DistributionError> {
            let mut zi_vec = zt[i].to_vec();
            let ave_zi = zi_vec.iter().sum::<f64>() / zi_vec.len() as f64;
            Ok(ave_zi)
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}
