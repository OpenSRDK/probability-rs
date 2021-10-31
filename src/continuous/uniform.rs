use crate::Distribution;
use rand::prelude::*;
use std::ops::Range;

#[derive(Clone, Debug)]
pub struct ContinuousUniform;

/// p returns the constant multiplied value so it can be used only for MCMC.
impl Distribution for ContinuousUniform {
    type T = f64;
    type U = Range<f64>;

    fn p(&self, _: &Self::T, theta: &Self::U) -> Result<f64, crate::DistributionError> {
        Ok(1.0 / (theta.end - theta.start))
    }

    fn sample(
        &self,
        theta: &Self::U,
        rng: &mut dyn RngCore,
    ) -> Result<Self::T, crate::DistributionError> {
        Ok(rng.gen_range(theta.clone()))
    }
}
