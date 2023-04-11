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
                            constantValue
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
            .fold(Expression::from(vec![0.0; theta_len]), |sum, x| sum + x)
            .assign(assignment);

        let result = if let Expression::Constant(value) = result_orig {
            let constantValue: ConstantValue = value;
            constantValue.into_tensor()
        }
        .to_vec();

        result
    }
}

#[cfg(test)]
mod tests {
    use opensrdk_kernel_method::RBF;
    use opensrdk_linear_algebra::{mat, SymmetricPackedMatrix};
    use opensrdk_symbolic_computation::{new_variable, Expression};
    use rand::{prelude::StdRng, Rng, SeedableRng};
    use rand_distr::StandardNormal;

    use crate::{
        ConditionMappableDistribution, ContinuousSamplesDistribution,
        DifferentiableConditionMappedDistribution, DistributionValueProduct, ExactEllipticalParams,
        MultivariateNormal, Normal, NormalParams,
    };

    use super::SteinVariationalGradientDescent;
    use opensrdk_kernel_method::*;

    #[test]
    fn it_works() {
        let mut rng = StdRng::from_seed([1; 32]);
        let mut rng2 = StdRng::from_seed([32; 32]);

        let samples_xy = (0..20)
            .into_iter()
            .map(|_| {
                let x = rng2.gen_range(-8.0..=8.0);
                let y = 1.0 + 0.5 * x + rng.sample::<f64, _>(StandardNormal);

                vec![x, y]
            })
            .collect::<Vec<Vec<f64>>>();

        let x = &samples_xy
            .iter()
            .map(|v| vec![1.0, v[0]])
            .collect::<Vec<_>>();
        let y = &samples_xy.iter().map(|v| v[1]).collect::<Vec<_>>();

        let sigma = Expression::from(0.5f64);

        let value = y.clone();

        let theta_0 = new_variable("alpha".to_owned());
        let theta_1 = new_variable("beta".to_owned());

        let likelihood = (1..x.len())
            .map(|i| {
                MultivariateNormal::new(
                    Expression::from(y[i]),
                    theta_0 * Expression::from(x[i][0]) + theta_1 * Expression::from(x[i][1]),
                    sigma,
                    1usize,
                )
            })
            .fold(
                MultivariateNormal::new(
                    Expression::from(y[0]),
                    theta_0 * Expression::from(x[0][0]) + theta_1 * Expression::from(x[0][1]),
                    sigma,
                    1usize,
                ),
                |sum, x| sum.mul(x),
            );

        // let dim = x[0].len();
        // let prior_sigma_sym =
        //     SymmetricPackedMatrix::from(dim, vec![0.5; dim * (dim + 1) / 2]).unwrap();
        // let prior_sigma = prior_sigma_sym.pptrf().unwrap();

        // let prior_mu = vec![0.5; dim];
        // let prior_params =
        //     ExactEllipticalParams::new(prior_mu.clone(), prior_sigma.clone()).unwrap();
        // let prior = MultivariateNormal::new().map_condition(|_| Ok(prior_params.clone()));

        // let kernel = RBF;
        // let kernel_params = [0.5, 0.5];
        // let samples_orig = (0..10)
        //     .into_iter()
        //     .map(|v| {
        //         let mut rng3 = StdRng::from_seed([v; 32]);
        //         let theta_0 = rng3.gen_range(-5.0..=5.0);
        //         let mut rng4 = StdRng::from_seed([v * 2; 32]);
        //         let theta_1 = rng4.gen_range(-5.0..=5.0);
        //         vec![theta_0, theta_1]
        //     })
        //     .collect::<Vec<Vec<f64>>>();
        // let samples = &mut ContinuousSamplesDistribution::new(samples_orig);

        // let theta = vec![0.1, 0.1];

        let stein_test = SteinVariationalGradientDescent::new(
            &likelihood,
            &prior,
            &kernel,
            &kernel_params,
            samples,
        );

        let phi = &stein_test.update_sample(&hash, 100f64);

        println!("{:?}", phi)
    }
}
