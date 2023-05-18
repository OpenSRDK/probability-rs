use crate::DiscreteDistribution;
use opensrdk_symbolic_computation::{Expression, Size};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bernoulli {
    k: Expression,
    p: Expression,
}

impl Bernoulli {
    pub fn new(k: Expression, p: Expression) -> Bernoulli {
        if k.mathematical_sizes() != vec![Size::Many, Size::One] && k.mathematical_sizes() != vec![]
        {
            panic!("k must be a scalar or a 2 rank vector");
        }
        Bernoulli { k, p }
    }
}

impl DiscreteDistribution for Bernoulli {
    fn value_ids(&self) -> HashSet<&str> {
        self.k.variable_ids()
    }

    fn conditions(&self) -> Vec<&Expression> {
        vec![&self.p]
    }

    fn pmf(&self) -> Expression {
        let k = self.k.clone();
        let p = self.p.clone();
        let pf_expression = p.clone().pow(k.clone()) * (1.0 - p).pow(1.0 - k);

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
