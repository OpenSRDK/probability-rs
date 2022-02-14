use opensrdk_linear_algebra::Vector;

use crate::{
    value, ConditionDifferentiableDistribution, DependentJoint, Distribution, RandomVariable,
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
        let diff_l = &self.lhs.ln_diff_value(&x.0, &x.1)?;
        let diff = (diff_l.clone().col_mat() * &self.rhs.fk(&x.1, theta)?).vec();
        Ok(diff)
    }
}

impl<L, R, T, UL, UR> ConditionDifferentiableDistribution for DependentJoint<L, R, T, UL, UR>
where
    L: Distribution<Value = T, Condition = UL> + ConditionDifferentiableDistribution,
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
        let diff_l = &self.lhs.ln_diff_condition(&x.0, &x.1)?;
        let diff_r = &self.rhs.ln_diff_condition(&x.1, &theta)?;
        let value_l = &self.lhs.fk(&x.0, &x.1)?;
        let value_r = &self.rhs.fk(&x.1, &theta)?;
        let diff = (diff_l.clone().col_mat() * value_r + diff_r.clone().col_mat() * value_l).vec();
        Ok(diff)
    }
}
