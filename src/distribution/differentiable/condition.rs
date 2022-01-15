use crate::Distribution;

pub trait ConditionDifferentiableDistribution: Distribution {
    fn ln_diff_condition(&self, x: &Self::Value, theta: &Self::Condition) -> Vec<f64>;
}
