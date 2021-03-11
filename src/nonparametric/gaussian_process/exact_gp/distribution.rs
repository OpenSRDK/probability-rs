use crate::{nonparametric::GaussianProcessParams, Distribution};
use crate::{MultivariateNormal, RandomVariable};
use opensrdk_kernel_method::*;
pub use rayon::prelude::*;
use std::{error::Error};

use super::ExactGP;

impl<K, T> Distribution for ExactGP<K, T>
where
    K: Kernel<T>,
    T: RandomVariable,
{
    type T = Vec<f64>;
    type U = GaussianProcessParams<T>;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, Box<dyn Error>> {
        let normal = MultivariateNormal;
        let params = self.handle_temporal_params(theta)?;

        return Ok(normal.p(x, &params)?);
    }

    fn sample(
        &self,
        theta: &Self::U,
        rng: &mut rand::prelude::StdRng,
    ) -> Result<Self::T, Box<dyn Error>> {
        let normal = MultivariateNormal;
        let params = self.handle_temporal_params(theta)?;

        return Ok(normal.sample(&params, rng)?);
    }
}
