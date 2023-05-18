use crate::{ContinuousDistribution, JointDistribution};
use opensrdk_symbolic_computation::{Expression, Size};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, f64::consts::PI, ops::Mul};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultivariateNormal {
    x: Expression,
    mu: Expression,
    sigma: Expression,
    d: usize,
}

impl MultivariateNormal {
    pub fn new(x: Expression, mu: Expression, sigma: Expression, d: usize) -> MultivariateNormal {
        if x.mathematical_sizes() != vec![Size::Many, Size::One] && x.mathematical_sizes() != vec![]
        {
            panic!("x must be a scalar or a 2 rank vector");
        }
        MultivariateNormal { x, mu, sigma, d }
    }
}

impl<Rhs> Mul<Rhs> for MultivariateNormal
where
    Rhs: ContinuousDistribution,
{
    type Output = JointDistribution<Self, Rhs>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        JointDistribution::new(self, rhs)
    }
}

impl ContinuousDistribution for MultivariateNormal {
    fn value_ids(&self) -> HashSet<&str> {
        self.x.variable_ids()
    }

    fn conditions(&self) -> Vec<&Expression> {
        vec![&self.mu, &self.sigma]
    }

    fn pdf(&self) -> Expression {
        let x = self.x.clone();
        let mu = self.mu.clone();
        let sigma = self.sigma.clone();
        let d = self.d as f64;

        let pdf_expression = (2.0 * PI).powf(-0.5 * d)
            * sigma.clone().det().pow((-0.5).into())
            * (-0.5
                * ((x.clone() - mu.clone())
                    .dot(sigma.inv(), &[[0, 0]])
                    .dot(x.clone() - mu.clone(), &[[1, 0]])))
            .exp();

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
