pub mod expected_improvement;
pub mod upper_confidence_bound;

pub use expected_improvement::*;
use ndarray::{Array, ArrayView1};
use opensrdk_kernel_method::{Periodic, PositiveDefiniteKernel, RBF};
use rand::{rngs::StdRng, Rng, SeedableRng};
pub use upper_confidence_bound::*;

use crate::{nonparametric::GaussianProcessRegressor, NormalParams};
use optimize::NelderMeadBuilder;

use super::BaseEllipticalProcessParams;

pub trait AcquisitionFunctions {
    fn value(&self, theta: &NormalParams) -> f64;
}

struct Data {
    x_data: Vec<Vec<f64>>,
    y_data: Vec<f64>,
}

#[test]
fn test_main() {
    let mut n: usize = 0;
    let mut data = Data {
        x_data: vec![],
        y_data: vec![],
    };

    loop {
        let mut rng = StdRng::from_seed([1; 32]);
        let mut x: f64 = rng.gen();

        sampling(&data, &x);

        n += 1;

        if n == 20 {
            break;
        }
    }

    loop {
        let xs = maximize_ucb(&data, n);
        // let xs = maximize_ei(&data);

        sampling(&data, &xs);

        n += 1;
    }
}

fn objective(x: &Vec<f64>) -> f64 {
    x + x.powf(2.0)
}

fn sampling(mut data: &Data, x: &Vec<f64>) {
    let y = objective(x);
    data.x_data.push(*x);
    data.y_data.push(y);
}

fn gp_regression(x: Vec<Vec<f64>>, y: &Vec<f64>, xs: &Vec<f64>) -> NormalParams {
    let kernel = RBF + Periodic;
    let theta = vec![1.0; kernel.params_len()];
    let sigma = 1.0;

    let base_params = BaseEllipticalProcessParams::new(kernel, x, theta, sigma).unwrap();
    let params_y = base_params.exact(y).unwrap();
    let mu = params_y.gp_predict(xs).unwrap().mu();
    let sigma = params_y.gp_predict(xs).unwrap().sigma();

    NormalParams { mu, sigma }
}

fn maximize_ucb(data: &Data, n: usize) -> f64 {
    let func_to_minimize = |xs: ArrayView1<f64>| {
        let theta: NormalParams = gp_regression(&data.x_data, &data.y_data, xs);
        let ucb = UpperConfidenceBound { trial: n };
        -ucb.value(&theta)
    };

    let minimizer = NelderMeadBuilder::default()
        .xtol(1e-6f64)
        .ftol(1e-6f64)
        .maxiter(50000)
        .build()
        .unwrap();

    // Set the starting guess
    let args = Array::from_vec(vec![3.0, -8.3]);
    let xs = minimizer.minimize(&func_to_minimize, args.view());

    xs
}

fn maximize_ei(data: &Data) -> f64 {
    let func_to_minimize = |xs: ArrayView1<f64>| {
        let theta: NormalParams = gp_regression(&data.x_data, &data.y_data, xs);
        let ei = ExpectedImprovement {
            f_vec: &data.y_data,
        };
        -ei.value(&theta)
    };

    let minimizer = NelderMeadBuilder::default()
        .xtol(1e-6f64)
        .ftol(1e-6f64)
        .maxiter(50000)
        .build()
        .unwrap();

    // Set the starting guess
    let args = Array::from_vec(vec![3.0, -8.3]);
    let xs = minimizer.minimize(&func_to_minimize, args.view());

    xs
}
