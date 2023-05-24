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
        if x.mathematical_sizes() != vec![Size::Many, Size::One]
            && x.mathematical_sizes() != vec![]
            && x.mathematical_sizes() != vec![Size::Many]
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

#[cfg(test)]
mod tests {
    use opensrdk_symbolic_computation::new_partial_variable;
    use opensrdk_symbolic_computation::new_variable;
    use opensrdk_symbolic_computation::opensrdk_linear_algebra::Matrix;
    use opensrdk_symbolic_computation::Expression;
    use opensrdk_symbolic_computation::ExpressionArray;

    use crate::ContinuousDistribution;
    use crate::MultivariateNormal;

    #[test]
    fn it_works() {
        let dim = 2usize;
        let theta_0 = new_variable("alpha".to_owned());
        let theta_1 = new_variable("beta".to_owned());

        let theta_vec = vec![theta_0.clone(), theta_1.clone()];
        let factory = |i: &[usize]| theta_vec[i[0].clone()].clone();
        let sizes: Vec<usize> = vec![theta_vec.len()];
        let theta_array_orig = ExpressionArray::from_factory(sizes, factory);
        let theta_array = new_partial_variable(theta_array_orig);
        let prior_sigma = Expression::from(Matrix::from(dim, vec![0.5; dim * dim]).unwrap());
        println!("{:?}", prior_sigma);

        let mat = Matrix::from(dim, vec![0.5, 0.5, 0.0, 0.5]).unwrap();
        let getrf = mat.getrf().unwrap();
        println!("{:?}", getrf);

        let prior_mu = Expression::from(vec![0.5; dim]);
        let prior = MultivariateNormal::new(theta_array, prior_mu, prior_sigma, dim);
        println!("{:?}", prior);
        //let result = prior.pdf();
        //println!("{:?}", result);
    }
}
