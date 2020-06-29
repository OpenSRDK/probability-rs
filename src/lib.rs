pub mod multinominal;
pub mod multivariate_normal;
pub mod normal;

use crate::linear_algebra::*;
use rand::prelude::*;

pub trait Distribution {
    fn sample(&self, thread_rng: &mut ThreadRng) -> f64;
}

pub trait MultivariateDistribution {
    fn sample(&self, thread_rng: &mut ThreadRng) -> Matrix;
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
