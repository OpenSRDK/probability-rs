pub mod expected_improvement;
pub mod upper_confidence_bound;

use cmaes::{fmax, DVector};
pub use expected_improvement::*;
use opensrdk_kernel_method::{Periodic, PositiveDefiniteKernel, RBF};
use rand::{rngs::StdRng, Rng, SeedableRng};
pub use upper_confidence_bound::*;

use crate::{nonparametric::GaussianProcessRegressor, NormalParams};

use super::BaseEllipticalProcessParams;

pub trait AcquisitionFunctions {
    fn value(&self, theta: &NormalParams) -> f64;
}

struct Data {
    x_history: Vec<Vec<f64>>,
    y_history: Vec<f64>,
}

#[test]
fn test_main() {
    let d: usize = 5; //データ数
    let mut n: usize = 0;
    let mut data = Data {
        x_history: vec![],
        y_history: vec![],
    };
    let mut k: usize = 0;

    loop {
        //x_1~x_nをサンプリング
        let mut x: Vec<f64> = vec![];

        loop {
            //x_n1~x_ndを生成してx_nにpush
            let mut rng = StdRng::from_seed([1; 32]);
            let r: f64 = rng.gen();
            x.push(r);

            k += 1;
            if k == d {
                break;
            }
        }

        sampling(&mut data, x);

        n += 1;

        if n == 20 {
            break;
        }
    }

    loop {
        let xs = vec![maximize_ucb(&data, n)];
        // let xs = maximize_ei(&data);

        sampling(&mut data, xs);

        n += 1;
    }
}

fn objective(_x: &Vec<f64>) -> f64 {
    // x + x.powf(2.0)
    todo!()
}

fn sampling(data: &mut Data, x: Vec<f64>) {
    let y = objective(&x);
    data.x_history.push(x);
    data.y_history.push(y);
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
    let func_to_maximize = |xs: Vec<f64>| {
        let theta: NormalParams = gp_regression(data.x_history, &data.y_history, &xs);
        let ucb = UpperConfidenceBound { trial: n };
        ucb.value(&theta)
    };

    let solution = fmax(
        |x: &DVector<f64>| func_to_maximize(x),
        data.x_history.to_vec(),
        0.01,
    );
    let xs = solution.point[0];
    xs
}

fn maximize_ei(data: &Data, n: usize) -> f64 {
    let func_to_maximize = |xs: Vec<f64>| {
        let theta: NormalParams = gp_regression(data.x_history, &data.y_history, &xs);
        let ei = ExpectedImprovement {
            f_vec: data.y_history,
        };
        ei.value(&theta)
    };

    let solution = fmax(
        |x: &DVector<f64>| func_to_maximize(x),
        data.x_history.to_vec(),
        0.01,
    );
    let xs = solution.point[0];
    xs
}
