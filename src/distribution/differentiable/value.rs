use crate::Distribution;

pub trait ValueDifferentiableDistribution: Distribution {
    fn ln_diff_value(&self, x: &Self::Value, theta: &Self::Condition) -> Vec<f64>;
}
