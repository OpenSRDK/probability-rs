use crate::{ContinuousDistribution, JointDistribution};
use opensrdk_symbolic_computation::{Expression, Size};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, f64::consts::PI, ops::Mul};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnivariateNormal {
    x: Expression,
    mu: Expression,
    sigma: Expression,
}

impl UnivariateNormal {
    pub fn new(x: Expression, mu: Expression, sigma: Expression) -> UnivariateNormal {
        if x.mathematical_sizes() != vec![Size::One, Size::One] && x.mathematical_sizes() != vec![]
        {
            panic!("x must be a scalar");
        }
        UnivariateNormal { x, mu, sigma }
    }
}

impl<Rhs> Mul<Rhs> for UnivariateNormal
where
    Rhs: ContinuousDistribution,
{
    type Output = JointDistribution<Self, Rhs>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        JointDistribution::new(self, rhs)
    }
}

impl ContinuousDistribution for UnivariateNormal {
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

        let pdf_expression = (1.0 / ((2.0 * PI).powf(0.5) * sigma.clone()))
            * -(((x - mu.clone()).pow(2.0.into())) / (2.0 * sigma.clone().pow(2.0.into())));
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

// #[cfg(test)]
// mod tests {
//     use crate::{
//         ConditionDifferentiableDistribution, Distribution, Normal, NormalParams,
//         SamplableDistribution, ValueDifferentiableDistribution,
//     };
//     use rand::prelude::*;

//     #[test]
//     fn it_works() {
//         let n = Normal;
//         let mut rng = StdRng::from_seed([1; 32]);

//         let mu = 2.0;
//         let sigma = 3.0;

//         let x = n
//             .sample(&NormalParams::new(mu, sigma).unwrap(), &mut rng)
//             .unwrap();

//         println!("{}", x);
//         let result = n.p_kernel(&0.5, &NormalParams::new(0.0, 1.0).unwrap());
//     }

//     #[test]
//     fn it_works2() {
//         let n = Normal;

//         let mu = 2.0;
//         let sigma = 3.0;

//         let x = 0.5;

//         let f = n.ln_diff_value(&x, &NormalParams::new(mu, sigma).unwrap());
//         println!("{:#?}", f);
//     }

//     #[test]
//     fn it_works_3() {
//         let n = Normal;

//         let mu = 0.0;
//         let sigma = 5.0;

//         let x = 1.0;

//         let f = n.ln_diff_condition(&x, &NormalParams::new(mu, sigma).unwrap());
//         println!("{:#?}", f);
//     }
// }
