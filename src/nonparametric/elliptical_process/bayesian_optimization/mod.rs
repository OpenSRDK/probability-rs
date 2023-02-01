pub mod expected_improvement;
pub mod upper_confidence_bound;

pub use expected_improvement::*;
use opensrdk_kernel_method::{Periodic, RBF};
pub use upper_confidence_bound::*;

use crate::NormalParams;

use super::BaseEllipticalProcessParams;

pub trait AcquisitionFunctions {
    fn value(&self, theta: &NormalParams) -> f64;
}

fn main() {
    //sampling
    let x = [1.0];
    let y = [1.0];
    let theta: NormalParams = gp_regression(x, y);
}
fn gp_regression(x: Vec<Vec<f64>>, y: Vec<Vec<f64>>) -> NormalParams {
    let kernel = RBF + Periodic;
    let theta = vec![1.0; kernel.params_len()];
    let sigma = 1.0;

    let base_params = BaseEllipticalProcessParams::new(kernel, x, theta, sigma).unwrap();
    let params_y = base_params.exact(y).unwrap();

    todo!()

    // [mu, var]
}
