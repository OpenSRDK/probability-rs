use crate::{ContinuousDistribution, JointDistribution};
use opensrdk_symbolic_computation::{ConstantValue, Expression, Size};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, ops::Mul};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExponentialDistribution {
    x: Expression,
    lambda: Expression,
}

impl ExponentialDistribution {
    pub fn new(x: Expression, lambda: Expression) -> ExponentialDistribution {
        if x.mathematical_sizes() != vec![Size::Many, Size::One] && x.mathematical_sizes() != vec![]
        {
            panic!("x must be a scalar or a 2 rank vector");
        }
        ExponentialDistribution { x, lambda }
    }
}

impl<Rhs> Mul<Rhs> for ExponentialDistribution
where
    Rhs: ContinuousDistribution,
{
    type Output = JointDistribution<Self, Rhs>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        JointDistribution::new(self, rhs)
    }
}

impl ContinuousDistribution for ExponentialDistribution {
    fn value_ids(&self) -> HashSet<&str> {
        self.x.variable_ids()
    }

    fn conditions(&self) -> Vec<&Expression> {
        vec![&self.lambda]
    }

    fn pdf(&self) -> Expression {
        let x = self.x.clone();

        if let Expression::Constant(value) = x.clone() {
            let constant_value: ConstantValue = value;
            if constant_value.into_scalar() < 0.0 {
                return 0.0.into();
            }
        }

        let lambda = self.lambda.clone();
        let pdf_expression = lambda.clone() * (-lambda * x).exp();

        pdf_expression
    }

    fn condition_ids(&self) -> HashSet<&str> {
        self.conditions()
            .iter()
            .map(|v| v.variable_ids())
            .flatten()
            .collect::<HashSet<_>>()
            .difference(&self.value_ids())
            .cloned()
            .collect()
    }

    fn ln_pdf(&self) -> Expression {
        self.pdf().ln()
    }
}
