use super::{ContinuousDistribution, JointDistribution};
use serde::Serialize;
use std::ops::Mul;

#[derive(Clone, Debug, Serialize)]
pub struct JointArrayDistribution<D> {
    distributions: Vec<D>,
}

pub trait DistributionProduct<D>
where
    D: ContinuousDistribution,
{
    /// p(x|a) = Π p(xi|ai)
    fn distribution_product(self) -> JointArrayDistribution<D>;
}

impl<I, D> DistributionProduct<D> for I
where
    I: Iterator<Item = D>,
    D: ContinuousDistribution,
{
    fn distribution_product(self) -> JointArrayDistribution<D> {
        let distributions = self.collect::<Vec<_>>();

        JointArrayDistribution { distributions }
    }
}

impl<D, Rhs> Mul<Rhs> for JointArrayDistribution<D>
where
    D: ContinuousDistribution,
    Rhs: ContinuousDistribution,
{
    type Output = JointDistribution<Self, Rhs>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        JointDistribution::new(self, rhs)
    }
}

impl<D> ContinuousDistribution for JointArrayDistribution<D>
where
    D: ContinuousDistribution,
{
    fn value_ids(&self) -> std::collections::HashSet<&str> {
        self.distributions
            .iter()
            .flat_map(|d| d.value_ids().into_iter())
            .collect()
    }

    fn conditions(&self) -> Vec<&opensrdk_symbolic_computation::Expression> {
        self.distributions
            .iter()
            .flat_map(|d| d.conditions().into_iter())
            .collect()
    }

    fn pdf(&self) -> opensrdk_symbolic_computation::Expression {
        self.distributions.iter().map(|d| d.pdf()).fold(
            opensrdk_symbolic_computation::Expression::from(1.0),
            |acc, x| acc * x,
        )
    }
}
