use crate::{Distribution, DistributionError};

pub trait ConditionDifferentiableDistribution: Distribution {
    fn ln_diff_condition(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError>;
}
