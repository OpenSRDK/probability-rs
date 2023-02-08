extern crate blas_src;
extern crate lapack_src;
extern crate opensrdk_kernel_method;
extern crate opensrdk_linear_algebra;
extern crate opensrdk_probability;
extern crate plotters;
extern crate rayon;

use crate::opensrdk_probability::*;
use opensrdk_kernel_method::*;
use opensrdk_linear_algebra::{pp::trf::PPTRF, Matrix};
use opensrdk_probability::{
    instant_value_differentiable::ValueDifferentiableInstantDistribution, nonparametric::*,
};
use plotters::{coord::Shift, prelude::*};
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::{iter::once, time::Instant};

#[derive(Clone, Debug)]
struct TensorGPParams {
    x: Matrix,
    theta: Vec<f64>,
    lc: PPTRF,
}

impl RandomVariable for TensorGPParams {
    type RestoreInfo = usize;

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        todo!()
    }

    fn len(&self) -> usize {
        todo!()
    }

    fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        todo!()
    }
}

#[test]
fn test_main() {
    let tensor_gp = ValueDifferentiableInstantDistribution::new(
        InstantDistribution::new(
            |(x, internal): &(Matrix, Vec<f64>), theta: &TensorGPParams| {
                let y = x;
                let yt = y.t();
                let sample_size = y.rows();
                let output_dim = y.cols();
                let x = theta.x;
                let xt = x.t();
                let lc = theta.lc;
                let theta = theta.theta;

                let inner_distr = (0..sample_size)
                    .map(|i| {
                        MultivariateNormal::new().map_differentiable_condition(
                            |v: &f64| ExactMultivariateNormalParams::new(vec![*v; output_dim], lc),
                            |v: &f64| todo!(),
                        )
                    })
                    .joint()
                    & MultivariateNormal::new().map_differentiable_condition(
                        |(x, theta): &(Matrix, Vec<f64>)| {
                            todo!();
                            ExactMultivariateNormalParams::new()
                        },
                        |(x, theta): &(Matrix, Vec<f64>)| todo!(),
                    );

                inner_distr.p_kernel(
                    &(
                        (0..sample_size).map(|i| yt[i].to_vec()).collect::<Vec<_>>(),
                        internal.clone(),
                    ),
                    &(x, theta),
                )
            },
            |_: &TensorGPParams, _: &mut RngCore| todo!(),
        ),
        |(x, internal): &(Matrix, Vec<f64>), _: &TensorGPParams| todo!(),
    );
    let rec_gp = ValueDifferentiableInstantDistribution::new(
        InstantDistribution::new(
            |(x, internal): &(Matrix, Vec<f64>), theta: &TensorGPParams| {
                let u = x;
                let ut = u.t();
                let sample_size = u.rows();
                let latent_dim = u.cols();
                let x = theta.x;
                let xt = x.t();
                let input_dim = x.cols();
                let lc = theta.lc;
                let theta = theta.theta;

                let inner_distr = (0..sample_size)
                    .map(|i| {
                        MultivariateNormal::new().map_differentiable_condition(
                            |v: &f64| ExactMultivariateNormalParams::new(vec![*v; latent_dim], lc),
                            |v: &f64| todo!(),
                        )
                    })
                    .joint()
                    & MultivariateNormal::new().map_differentiable_condition(
                        |(x, theta): &(Matrix, Vec<f64>)| {
                            let input = Matrix::from(
                                latent_dim + input_dim,
                                once(vec![0.0; latent_dim].iter())
                                    .chain((0..sample_size - 1).map(|i| ut[i].iter()))
                                    .zip((0..sample_size).map(|i| xt[i].iter()))
                                    .flat_map(|(ui, xi)| ui.chain(xi))
                                    .map(|value| *value)
                                    .collect::<Vec<_>>(),
                            )
                            .unwrap()
                            .t();

                            ExactMultivariateNormalParams::new()
                        },
                        |(x, theta): &(Matrix, Vec<f64>)| todo!(),
                    );

                inner_distr.p_kernel(
                    &(
                        (0..sample_size).map(|i| ut[i].to_vec()).collect::<Vec<_>>(),
                        internal.clone(),
                    ),
                    &(x, theta),
                )
            },
            |_: &TensorGPParams, _: &mut RngCore| todo!(),
        ),
        |(x, internal): &(Matrix, Vec<f64>), _: &TensorGPParams| todo!(),
    );
}
