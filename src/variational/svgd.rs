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
                    .unwrap()
                    .assign(assignment);
                let kernel_diff = self
                    .kernel
                    .expression(&theta_vec, samples_array, self.kernel_params)
                    .unwrap()
                    .ln()
                    .differential(self.kernel.value_ids())
                    .iter()
                    .map(|i| i.assign(assignment))
                    .collect::<Vec<Expression>>();
                let p_diff_lhs = self
                    .likelihood
                    .pdf()
                    .ln()
                    .differential(self.likelihood.condition_ids())
                    .iter()
                    .map(|i| i.assign(assignment))
                    .collect::<Vec<Expression>>();
                let p_diff_rhs = self
                    .prior
                    .pdf()
                    .ln()
                    .differential(self.prior.value_ids())
                    .iter()
                    .map(|i| i.assign(assignment))
                    .collect::<Vec<Expression>>();
                let result = kernel_diff
                    .iter()
                    .enumerate()
                    .map(|(i, kernel_diff_i)| {
                        (kernel_diff_i + kernel * (p_diff_lhs[i] + p_diff_rhs[i])).into_scalar()
                    })
                    .collect::<Vec<f64>>();
                result
            })
            .fold(vec![0.0; m], |sum, x| {
                sum.iter()
                    .enumerate()
                    .map(|(i, sum_i)| sum_i + x[i])
                    .collect::<Vec<f64>>()
            });

        let phi = phi_sum.iter().map(|i| i / n as f64).collect::<Vec<f64>>();
        Ok(phi)
    }

    pub fn update_sample(
        &self,
        assignment: &HashMap<&str, ConstantValue>,
        step_size: f64,
    ) -> Vec<f64> {
        let samples_len = self.samples.len();
        let mut phi = vec![0.0; samples_len];
        let epsilon = 0.0001;
        let stein_mut = &mut SteinVariationalGradientDescent::new(
            self.likelihood,
            self.prior,
            self.kernel,
            self.kernel_params,
            self.samples,
        );
        for i in 0..step_size {
            let direction = stein_mut.direction(assignment);
            let samples_new = stein_mut
                .samples
                .iter()
                .zip(phi.iter())
                .map(|(theta_i, phi_i)| theta_i + phi_i * &epsilon)
                .collect::<Vec<f64>>();
            stein_mut = &mut SteinVariationalGradientDescent::new(
                self.likelihood,
                self.prior,
                self.kernel,
                self.kernel_params,
                samples_new,
            );
        }
    }
}
