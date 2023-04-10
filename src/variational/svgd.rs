use std::{collections::HashMap, ptr::hash};

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
        let sizes: Vec<usize> = (0usize..theta_vec.len()).collect();
        let theta_array_orig = ExpressionArray::from_factory(sizes, factory);
        let theta_array = new_partial_variable(theta_array_orig);

        let phi_sum = self
            .samples
            .iter()
            .map(|theta_j| {
                let samples_array = new_partial_variable(theta_j);
                let kernel = self
                    .kernel
                    .expression(theta_array, samples_array, self.kernel_params)
                    .unwrap()
                    .assign(assignment);
                let kernel_diff = self
                    .kernel
                    .expression(theta_array, samples_array, self.kernel_params)
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
                        let expression: Expression =
                            (kernel_diff_i + kernel * (p_diff_lhs[i] + p_diff_rhs[i]));

                        if let Expression::Constant(value) = expression {
                            let constantValue: ConstantValue = value;
                            constantValue.into_scalar()
                        }
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

        // Assign parameters which is not estimated
        //let likelihood_assign = self.likelihood.expression().assign(assignment);
        //let prior_assign = self.prior.expression().assign(assignment);

        let stein_mut = &mut SteinVariationalGradientDescent::new(
            self.likelihood,
            self.prior,
            self.kernel,
            self.kernel_params,
            self.samples,
        );

        let str = self.likelihood.condition_ids();
        let str_vec: Vec<_> = str.into_iter().collect();

        let theta_len = str_vec.len();

        for i in 0..step_size {
            let samples_new = stein_mut
                .samples
                .iter()
                .map(|theta| {
                    let theta_map = HashMap::new();
                    let len = theta.elems().len();

                    for i in 0..len {
                        let expression = theta.elems().get(&vec![i]).unwrap();

                        let elem = if let Expression::Constant(value) = expression {
                            let constantValue: ConstantValue = value;
                            constantValue //.into_scalar()
                        };

                        theta_map.insert(str_vec[i], elem)
                    }

                    phi = stein_mut.direction(&theta_map).unwrap();

                    let theta_new = theta
                        .elems()
                        .iter()
                        .zip(phi.iter())
                        .map(|(theta_i, phi_i)| {
                            let theta_i_elem = theta_i.1;
                            let elem = *theta_i_elem
                                + Expression::from(phi_i) * Expression::from(&epsilon);
                            let elem_assign = elem.assign(assignment);
                            elem
                        })
                        .collect::<Vec<Expression>>();

                    let factory = |i: &[usize]| theta_new[i[0].clone()].clone();
                    let sizes: Vec<usize> = (0usize..theta_new.len()).collect();
                    let theta_new_array = ExpressionArray::from_factory(sizes, factory);
                    theta_new_array
                })
                .collect::<Vec<ExpressionArray>>();

            stein_mut = &mut SteinVariationalGradientDescent::new(
                self.likelihood,
                self.prior,
                self.kernel,
                self.kernel_params,
                samples_new,
            );
        }
        let result_orig = stein_mut
            .samples
            .iter()
            .map(|result_array_orig| {
                let result_array = new_partial_variable(result_array_orig);
                result_array
            })
            .fold(Expression::from(vec![0.0; theta_len]), |sum, x| sum + x);

        let result = if let Expression::Constant(value) = result_orig {
            let constantValue: ConstantValue = value;
            constantValue.into_tensor()
        }
        .to_vec();
        result
    }
}
