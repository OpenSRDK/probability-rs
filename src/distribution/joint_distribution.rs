use std::ops::Mul;

use super::ContinuousDistribution;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct JointDistribution<Lhs, Rhs>
where
    Lhs: ContinuousDistribution,
    Rhs: ContinuousDistribution,
{
    dl: Lhs,
    dr: Rhs,
}

impl<Lhs, Rhs> JointDistribution<Lhs, Rhs>
where
    Lhs: ContinuousDistribution,
    Rhs: ContinuousDistribution,
{
    pub fn new(dl: Lhs, dr: Rhs) -> JointDistribution<Lhs, Rhs> {
        JointDistribution { dl, dr }
    }
}

impl<SelfLhs, SelfRhs, Rhs> Mul<Rhs> for JointDistribution<SelfLhs, SelfRhs>
where
    SelfLhs: ContinuousDistribution,
    SelfRhs: ContinuousDistribution,
    Rhs: ContinuousDistribution,
{
    type Output = JointDistribution<Self, Rhs>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        JointDistribution::new(self, rhs)
    }
}

impl<Lhs, Rhs> ContinuousDistribution for JointDistribution<Lhs, Rhs>
where
    Lhs: ContinuousDistribution,
    Rhs: ContinuousDistribution,
{
    fn value_ids(&self) -> std::collections::HashSet<&str> {
        self.dl
            .value_ids()
            .into_iter()
            .chain(self.dr.value_ids().into_iter())
            .collect()
    }

    fn conditions(&self) -> Vec<&opensrdk_symbolic_computation::Expression> {
        self.dl
            .conditions()
            .into_iter()
            .chain(self.dr.conditions().into_iter())
            .collect()
    }

    fn pdf(&self) -> opensrdk_symbolic_computation::Expression {
        self.dl.pdf() * self.dr.pdf()
    }
}
