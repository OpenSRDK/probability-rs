use crate::{ContinuousDistribution, JointDistribution};
use opensrdk_symbolic_computation::{ConstantValue, Expression, Size};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, ops::Mul};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Uniform {
    x: Expression,
    a: Expression,
    b: Expression,
}

impl Uniform {
    pub fn new(x: Expression, a: Expression, b: Expression) -> Uniform {
        if x.mathematical_sizes() != vec![Size::Many, Size::One] && x.mathematical_sizes() != vec![]
        {
            panic!("x must be a scalar or a 2 rank vector");
        }
        Uniform { x, a, b }
    }
}

impl<Rhs> Mul<Rhs> for Uniform
where
    Rhs: ContinuousDistribution,
{
    type Output = JointDistribution<Self, Rhs>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        JointDistribution::new(self, rhs)
    }
}

impl ContinuousDistribution for Uniform {
    fn value_ids(&self) -> HashSet<&str> {
        self.x.variable_ids()
    }

    fn conditions(&self) -> Vec<&Expression> {
        vec![&self.a, &self.b]
    }

    fn pdf(&self) -> Expression {
        let a = self.a.clone();
        let b = self.b.clone();

        let pdf_expression =
            Expression::Constant(ConstantValue::Scalar(1.0)) / (b.clone() - a.clone());
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
