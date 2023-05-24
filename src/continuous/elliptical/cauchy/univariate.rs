use crate::{ContinuousDistribution, JointDistribution};
use opensrdk_symbolic_computation::{Expression, Size};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, f64::consts::PI, ops::Mul};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnivariateCauchy {
    x: Expression,
    x0: Expression,
    gamma: Expression,
}

impl UnivariateCauchy {
    pub fn new(x: Expression, x0: Expression, gamma: Expression) -> UnivariateCauchy {
        if x.mathematical_sizes() != vec![Size::One, Size::One] && x.mathematical_sizes() != vec![]
        {
            panic!("x must be a scalar");
        }
        UnivariateCauchy { x, x0, gamma }
    }
}

impl<Rhs> Mul<Rhs> for UnivariateCauchy
where
    Rhs: ContinuousDistribution,
{
    type Output = JointDistribution<Self, Rhs>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        JointDistribution::new(self, rhs)
    }
}

impl ContinuousDistribution for UnivariateCauchy {
    fn value_ids(&self) -> HashSet<&str> {
        self.x.variable_ids()
    }

    fn conditions(&self) -> Vec<&Expression> {
        vec![&self.x0, &self.gamma]
    }

    fn pdf(&self) -> Expression {
        let x = self.x.clone();
        let x0 = self.x0.clone();
        let gamma = self.gamma.clone();

        let pdf_expression =
            gamma.clone() / (PI * (gamma.pow(2.0.into()) + (x - x0).pow(2.0.into())));

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
//         ConditionDifferentiableDistribution, Distribution, ExactMultivariateCauchyParams,
//         MultivariateCauchy, SamplableDistribution, ValueDifferentiableDistribution,
//     };
//     use opensrdk_linear_algebra::{pp::trf::PPTRF, *};
//     use rand::prelude::*;
//     #[test]
//     fn it_works() {
//         let cauchy = MultivariateCauchy::new();
//         let mut rng = StdRng::from_seed([1; 32]);

//         let mu = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
//         let lsigma = SymmetricPackedMatrix::from_mat(&mat!(
//            1.0,  0.0,  0.0,  0.0,  0.0,  0.0;
//            2.0,  3.0,  0.0,  0.0,  0.0,  0.0;
//            4.0,  5.0,  6.0,  0.0,  0.0,  0.0;
//            7.0,  8.0,  9.0, 10.0,  0.0,  0.0;
//           11.0, 12.0, 13.0, 14.0, 15.0,  0.0;
//           16.0, 17.0, 18.0, 19.0, 20.0, 21.0
//         ))
//         .unwrap();
//         println!("{:#?}", lsigma);

//         let x = cauchy
//             .sample(
//                 &ExactMultivariateCauchyParams::new(mu, PPTRF(lsigma)).unwrap(),
//                 &mut rng,
//             )
//             .unwrap();

//         println!("{:#?}", x);
//     }

//     #[test]
//     fn it_works2() {
//         let cauchy = MultivariateCauchy::new();

//         let mu = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
//         let lsigma = SymmetricPackedMatrix::from_mat(&mat!(
//            1.0,  0.0,  0.0,  0.0,  0.0,  0.0;
//            2.0,  3.0,  0.0,  0.0,  0.0,  0.0;
//            4.0,  5.0,  6.0,  0.0,  0.0,  0.0;
//            7.0,  8.0,  9.0, 10.0,  0.0,  0.0;
//           11.0, 12.0, 13.0, 14.0, 15.0,  0.0;
//           16.0, 17.0, 18.0, 19.0, 20.0, 21.0
//         ))
//         .unwrap();
//         let x = vec![0.0, 1.0, 2.0, 2.0, 2.0, 4.0];

//         let f = cauchy.ln_diff_value(
//             &x,
//             &ExactMultivariateCauchyParams::new(mu, PPTRF(lsigma)).unwrap(),
//         );
//         println!("{:#?}", f);
//     }

//     #[test]
//     fn it_works_3() {
//         let cauchy = MultivariateCauchy::new();

//         let mu = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
//         let lsigma = SymmetricPackedMatrix::from_mat(&mat!(
//            1.0,  0.0,  0.0,  0.0,  0.0,  0.0;
//            2.0,  3.0,  0.0,  0.0,  0.0,  0.0;
//            4.0,  5.0,  6.0,  0.0,  0.0,  0.0;
//            7.0,  8.0,  9.0, 10.0,  0.0,  0.0;
//           11.0, 12.0, 13.0, 14.0, 15.0,  0.0;
//           16.0, 17.0, 18.0, 19.0, 20.0, 21.0
//         ))
//         .unwrap();
//         let x = vec![0.0, 1.0, 2.0, 2.0, 2.0, 4.0];

//         let f = cauchy.ln_diff_condition(
//             &x,
//             &ExactMultivariateCauchyParams::new(mu, PPTRF(lsigma)).unwrap(),
//         );
//         println!("{:#?}", f);
//     }
// }
