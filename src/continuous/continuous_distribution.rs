use std::{collections::HashSet, fmt::Debug};

use opensrdk_symbolic_computation::Expression;
use serde::{Deserialize, Serialize};

pub trait ContinuousDistribution: Clone + Debug + Serialize {
    fn value_ids(&self) -> HashSet<&str>;

    fn conditions(&self) -> Vec<&Expression>;

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

    fn pdf(&self) -> Expression;

    fn ln_pdf(&self) -> Expression {
        self.pdf().ln()
    }
}
