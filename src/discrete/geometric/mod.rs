use crate::{DiscreteDistribution, JointDistribution};
use opensrdk_symbolic_computation::{Expression, Size};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, ops::Mul};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Geometric {
    k: Expression,
    p: Expression,
}

impl Geometric {
    pub fn new(k: Expression, p: Expression) -> Geometric {
        if k.mathematical_sizes() != vec![Size::Many, Size::One] {
            panic!("k must be a scalar or a 2 rank vector");
        }
        Geometric { k, p }
    }
}

// impl<Rhs> Mul<Rhs> for Geometric
// where
//     Rhs: DiscreteDistribution,
// {
//     type Output = JointDistribution<Self, Rhs>;

//     fn mul(self, rhs: Rhs) -> Self::Output {
//         JointDistribution::new(self, rhs)
//     }
// }

impl DiscreteDistribution for Geometric {
    fn value_ids(&self) -> HashSet<&str> {
        self.k.variable_ids()
    }

    fn conditions(&self) -> Vec<&Expression> {
        vec![&self.p]
    }

    fn pmf(&self) -> Expression {
        let k = self.k.clone();
        let p = self.p.clone();
        let pf_expression = (1.0 - p).pow(k - 1.0) * p;

        pf_expression
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

    fn ln_pmf(&self) -> Expression {
        self.pmf().ln()
    }
}
