use std::collections::HashMap;

use opensrdk_kernel_method::PositiveDefiniteKernel;
use opensrdk_symbolic_computation::{
    new_partial_variable, ConstantValue, Expression, ExpressionArray,
};

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
    samples: Vec<ExpressionArray>,
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
        samples: Vec<ExpressionArray>,
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
        let theta_vec = self.likelihood.conditions().clone();
        let factory = |i: &[usize]| theta_vec[i[0].clone()].clone();
        let sizes = vec![1usize; theta_vec.len()];
        let theta_array = theta_vec.iter().index();
        let expression_orig = ExpressionArray::new(vec![0.0; m]);
        let phi_sum = self
            .samples
            .iter()
            .map(|theta_j| {
                let samples_array = new_partial_variable(theta_j);
                let kernel = self
                    .kernel
                    .expression(&theta_vec, samples_array, self.kernel_params)
                    .unwrap();
                let kernel_diff = self
                    .kernel
                    .expression(&theta_vec, samples_array, self.kernel_params)
                    .unwrap()
                    .ln()
                    .differential(self.kernel.value_ids()); //variable_ids is value;
                let p_diff = self
                    .likelihood
                    .pdf()
                    .ln()
                    .differential(self.likelihood.condition_ids()) //variable_ids is condition
                    + self
                        .prior
                        .pdf()
                        .ln()
                        .differential(self.prior.value_ids()); //variable_ids is value;
                let expression_result = kernel * p_diff + kernel_diff;
                result
            })
            .fold(expression_orig, |sum, x| sum.index() + x);

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
