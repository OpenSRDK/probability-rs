extern crate blas_src;
extern crate lapack_src;
extern crate opensrdk_kernel_method;
extern crate opensrdk_linear_algebra;
extern crate opensrdk_probability;
extern crate plotters;
extern crate rayon;

use crate::opensrdk_probability::*;
use opensrdk_kernel_method::*;
use opensrdk_linear_algebra::{mat, DiagonalMatrix, Matrix, Vector};
use opensrdk_probability::nonparametric::*;
use plotters::{coord::Shift, prelude::*};
use rand::{prelude::*, seq::index::sample};
use rand_distr::StandardNormal;
use std::{convert::identity, time::Instant};

#[test]
fn test_main() {}

fn fk(x: Matrix, y: Vec<f64>) -> Result<(f64, Vec<f64>), DistributionError> {
    let n = x.cols();

    let w_sigma = DiagonalMatrix::identity(n).mat();
    let w_mu = vec![0.0; n];
    let w_params = |_: _| ExactEllipticalParams::new(w_mu, w_sigma);
    let prior_distr = MultivariateNormal::new().condition(w_params);

    let distr_y = (0..x.rows())
        .into_iter()
        .map(|i| -> Result<_, DistributionError> {
            let xi = x.eject_row(i).row_mat();
            let distr_yi = Normal.condition(|&params: &Vec<f64>| {
                NormalParams::new(
                    params.split_off(1).row_mat().linear_prod(&xi) + params[0],
                    1.0,
                )
            });
            Ok(distr_yi)
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let poster_distr = distr_y.into_iter().joint();

    let alpha0 = vec![0.0];
    let beta0 = vec![0.5; n];
    let params0 = alpha0.append(&mut beta0);

    let sampler = EllipticalSliceSampler::new(params0, &poster_distr, &prior_distr);
    let mut rng = StdRng::from_seed([1; 32]);
    let params_sample = sampler.sample(rng);

    Ok(())
}
