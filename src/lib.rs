extern crate opensrdk_linear_algebra;
extern crate rand;
extern crate rand_distr;

pub mod multivariate_normal;
pub mod normal;
pub mod prelude;

use opensrdk_linear_algebra::prelude::*;
use rand::prelude::*;

pub trait Distribution {
    fn sample(&self, thread_rng: &mut ThreadRng) -> Result<f64, String>;
}

pub trait MultivariateDistribution {
    fn sample(&self, thread_rng: &mut ThreadRng) -> Result<Matrix, String>;
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
