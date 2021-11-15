use crate::Distribution;
use rand::prelude::*;
use std::ops::Range;

#[derive(Clone, Debug)]
pub struct ContinuousUniform;

/// p returns the constant multiplied value so it can be used only for MCMC.
impl Distribution for ContinuousUniform {
    type Value = f64;
    type Condition = Range<f64>;

    fn fk(
        &self,
        _: &Self::Value,
        _theta: &Self::Condition,
    ) -> Result<f64, crate::DistributionError> {
        Ok(1.0)
    }

    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, crate::DistributionError> {
        Ok(rng.gen_range(theta.clone()))
    }
}
