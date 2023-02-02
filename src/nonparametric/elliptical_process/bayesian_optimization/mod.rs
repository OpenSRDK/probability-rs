pub mod expected_improvement;
pub mod upper_confidence_bound;

pub use expected_improvement::*;
use opensrdk_kernel_method::{Periodic, RBF};
use rand::{rngs::StdRng, Rng, RngCore};
pub use upper_confidence_bound::*;

use crate::{nonparametric::GaussianProcessRegressor, NormalParams};

use super::BaseEllipticalProcessParams;

pub trait AcquisitionFunctions {
    fn value(&self, theta: &NormalParams) -> f64;
}

struct Data {
    x_data: Vec<f64>,
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

    let theta: NormalParams = gp_regression(&data.x_data, &data.y_data);
    //let xs = (UCB(mu,sigma)を最大化するx);
}

fn objective(x: &f64) -> f64 {
    x + x ^ 2.0
}

fn sampling(mut data: &Data, x: &f64) {
    let y = objective(x);
    data.x_data.push(x);
    data.y_data.push(y);
}

fn gp_regression(x: &Vec<f64>, y: &Vec<f64>) -> NormalParams {
    let kernel = RBF + Periodic;
    let theta = vec![1.0; kernel.params_len()];
    let sigma = 1.0;

    //これで出るmuとsigmaを用いて計算されるUCBが最大になるxが次のx_n+1.だからこれではダメで、これを最大化するようなxを求められるようにする.
    let x = x[0..x.len() - 1];
    let xs = x[x.len()];
    let y = x[0..x.len() - 1];

    let base_params = BaseEllipticalProcessParams::new(kernel, x, theta, sigma).unwrap();
    let params_y = base_params.exact(&y).unwrap();
    let mu = params_y.gp_predict(&xs).unwrap().mu();
    let sigma = params_y.gp_predict(&xs).unwrap().sigma();

    [mu, sigma]
}
