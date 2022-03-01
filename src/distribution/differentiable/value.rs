use crate::{Distribution, DistributionError, RandomVariable};

pub trait ValueDifferentiableDistribution: Distribution {
    fn ln_diff_value(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError>;
}
