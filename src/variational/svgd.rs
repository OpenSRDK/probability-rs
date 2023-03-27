use std::collections::HashMap;

use opensrdk_kernel_method::PositiveDefiniteKernel;
use opensrdk_symbolic_computation::{ConstantValue, Expression};

use crate::ContinuousDistribution;

pub struct SteinVariationalGradientDescent<'a, D, K>
where
    D: ContinuousDistribution,
    K: PositiveDefiniteKernel,
{
    likelihood: &'a D,
    prior: &'a D,
    kernel: &'a K,
    kernel_params: &'a [f64],
    samples: Vec<Expression>,
}

impl<'a, D, K> SteinVariationalGradientDescent<'a, D, K>
where
    D: ContinuousDistribution,
    K: PositiveDefiniteKernel,
{
    pub fn new(
        likelihood: &'a D,
        prior: &'a D,
        kernel: &'a K,
        kernel_params: &'a [f64],
        samples: Vec<Expression>,
    ) -> Self {
        Self {
            likelihood,
            prior,
            kernel,
            kernel_params,
            samples,
        }
    }

    pub fn direction(&self, assignment: &HashMap<&str, ConstantValue>) -> Vec<f64> {
        let n = self.samples.samples().len();
        let m = self.samples.samples()[0].len();
        let theta_vec = self.likelihood.theta.clone().transform_vec().0;
        let phi_sum = self
            .samples
            .iter()
            .map(|theta_j| {
                let kernel = self
                    .kernel
                    .expression(&theta_vec, &theta_j, self.kernel_params)
                    .unwrap();
                let kernel_diff = self
                    .kernel
                    .expession(&theta_vec, &theta_j, self.kernel_params)
                    .ln()
                    .differential() //variable_ids is value
                    .unwrap();
                let p_diff = self
                    .likelihood
                    .pdf()
                    .ln()
                    .differential(&["mu", "sigma"]) //variable_ids is condition
                    .unwrap()
                    + self
                        .prior
                        .pdf()
                        .ln()
                        .differential(&["mu", "sigma"]) //variable_ids is value
                        .unwrap();
                kernel * p_diff + kernel_diff
            })
            .fold(vec![0.0; m].col_mat(), |sum, x| sum + x);
        let phi = phi_sum
            .vec()
            .iter()
            .map(|i| i / n as f64)
            .collect::<Vec<Expression>>();
        let result = phi.assign(assignment);
        Ok(result)
    }

    pub fn update_sample(
        &self,
        assignment: &HashMap<&str, ConstantValue>,
        step_size: f64,
    ) -> Vec<f64> {
        let direction = self.direction(assignment);
        todo!()
    }
}
