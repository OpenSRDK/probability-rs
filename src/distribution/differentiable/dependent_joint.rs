use crate::{
    ConditionDifferentiableDistribution, DependentJoint, Distribution, RandomVariable,
    ValueDifferentiableDistribution,
};

impl<L, R, T, UL, UR> ValueDifferentiableDistribution for DependentJoint<L, R, T, UL, UR>
where
    L: Distribution<Value = T, Condition = UL>
        + ValueDifferentiableDistribution
        + ConditionDifferentiableDistribution,
    R: Distribution<Value = UL, Condition = UR> + ValueDifferentiableDistribution,
    T: RandomVariable,
    UL: RandomVariable,
    UR: RandomVariable,
{
    fn ln_diff_value(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, crate::DistributionError> {
        todo!()
    }
}

impl<L, R, T, UL, UR> ConditionDifferentiableDistribution for DependentJoint<L, R, T, UL, UR>
where
    L: Distribution<Value = T, Condition = UL>,
    R: Distribution<Value = UL, Condition = UR> + ConditionDifferentiableDistribution,
    T: RandomVariable,
    UL: RandomVariable,
    UR: RandomVariable,
{
    fn ln_diff_condition(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, crate::DistributionError> {
        todo!()
    }
}
