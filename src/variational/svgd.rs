use std::{collections::HashMap, ptr::hash};

use opensrdk_kernel_method::{Constant, PositiveDefiniteKernel};
use opensrdk_symbolic_computation::{
    new_partial_variable, new_variable, ConstantValue, Expression, ExpressionArray,
};

use crate::ContinuousDistribution;

pub struct SteinVariationalGradientDescent<'a, D, P, K>
where
    D: ContinuousDistribution,
    P: ContinuousDistribution,
    K: PositiveDefiniteKernel,
{
    likelihood: &'a D,
    prior: &'a P,
    kernel: &'a K,
    kernel_params: &'a [f64],
    samples: Vec<ExpressionArray>,
}

impl<'a, D, P, K> SteinVariationalGradientDescent<'a, D, P, K>
where
    D: ContinuousDistribution,
    P: ContinuousDistribution,
    K: PositiveDefiniteKernel,
{
    pub fn new(
        likelihood: &'a D,
        prior: &'a P,
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
        let n = self.samples.len();
        let m = self.samples[0].elems().len();

        //let theta_vec = self.likelihood.conditions().clone();
        let theta_vec = vec![
            new_variable("alpha".to_owned()),
            new_variable("beta".to_owned()),
        ];
        println!("{:?}", theta_vec);
        let factory = |i: &[usize]| theta_vec[i[0].clone()].clone();
        let sizes: Vec<usize> = vec![theta_vec.len()];
        println!("{:?}", "two point five");
        let theta_array_orig = ExpressionArray::from_factory(sizes, factory);
        let theta_array = new_partial_variable(theta_array_orig);
        println!("{:?}", "three");

        let kernel_params_expression = self
            .kernel_params
            .iter()
            .map(|elem| Expression::from(*elem))
            .collect::<Vec<Expression>>();
        println!("{:?}", "four");

        let theta_ids = &self
            .likelihood
            .condition_ids()
            .iter()
            .map(|i| *i)
            .collect::<Vec<_>>();
        println!("{:?}", "five");

        let samples_array = new_partial_variable(self.samples[0].clone());
        println!("{:?}", samples_array);
        println!("{:?}", theta_array);
        println!("{:?}", kernel_params_expression);

        let phi_sum = self
            .samples
            .iter()
            .map(|theta_j| {
                let samples_array = new_partial_variable(theta_j.clone());
                let p_diff_rhs = self
                    .prior
                    .pdf()
                    .ln()
                    .differential(theta_ids)
                    .iter()
                    .map(|i| i.clone().assign(assignment))
                    .collect::<Vec<Expression>>();
                let p_diff_lhs = self
                    .likelihood
                    .pdf()
                    .ln()
                    .differential(theta_ids)
                    .iter()
                    .map(|i| i.clone().assign(assignment))
                    .collect::<Vec<Expression>>();
                let kernel = self
                    .kernel
                    .expression(
                        theta_array.clone(),
                        samples_array.clone(),
                        &kernel_params_expression,
                    )
                    .unwrap()
                    .assign(assignment);
                let kernel_diff = self
                    .kernel
                    .expression(
                        theta_array.clone(),
                        samples_array.clone(),
                        &kernel_params_expression,
                    )
                    .unwrap()
                    .ln()
                    .differential(theta_ids)
                    .iter()
                    .map(|i| i.clone().assign(assignment))
                    .collect::<Vec<Expression>>();
                println!("{:?}", p_diff_lhs[0]);
                println!("{:?}", p_diff_rhs[0]);
                println!("{:?}", kernel_diff[0]);
                let result = kernel_diff
                    .iter()
                    .enumerate()
                    .map(|(i, kernel_diff_i)| {
                        let expression: Expression = (kernel_diff_i.clone()
                            + kernel.clone() * (p_diff_lhs[i].clone() + p_diff_rhs[i].clone()));

                        let result_elem: ConstantValue = expression.into();
                        result_elem.into_scalar()
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
        println!("{:?}", "six");

        let phi = phi_sum.iter().map(|i| i / n as f64).collect::<Vec<f64>>();
        phi
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
            self.samples.clone(),
        );

        let str = self.likelihood.condition_ids();
        let str_vec: Vec<_> = str.into_iter().collect();

        let theta_len = str_vec.len();

        for i in 0..step_size as usize {
            let samples_new = stein_mut
                .samples
                .iter()
                .map(|theta| {
                    let theta_map = &mut HashMap::new();
                    let len = theta.elems().len();

                    for i in 0..len {
                        let expression = theta.elems().get(&vec![i]).unwrap();

                        let elem: ConstantValue = expression.clone().into();

                        theta_map.insert(str_vec[i], elem);
                    }

                    phi = stein_mut.direction(&theta_map);

                    let theta_new = theta
                        .elems()
                        .iter()
                        .zip(phi.iter())
                        .map(|(theta_i, phi_i)| {
                            let theta_i_elem = theta_i.1;
                            let elem = theta_i_elem.clone()
                                + Expression::from(phi_i.clone()) * Expression::from(epsilon);
                            let elem_assign = elem.assign(assignment);
                            elem_assign
                        })
                        .collect::<Vec<Expression>>();

                    let factory = |i: &[usize]| theta_new[i[0].clone()].clone();
                    let sizes: Vec<usize> = (0usize..theta_new.len()).collect();
                    let theta_new_array = ExpressionArray::from_factory(sizes, factory);
                    theta_new_array
                })
                .collect::<Vec<ExpressionArray>>();

            let stein_mut = &mut SteinVariationalGradientDescent::new(
                self.likelihood,
                self.prior,
                self.kernel,
                self.kernel_params,
                samples_new,
            );
            println!("{:?}", i);
        }
        let result_orig = stein_mut
            .samples
            .iter()
            .map(|result_array_orig| {
                let result_array = new_partial_variable(result_array_orig.clone());
                result_array
            })
            .fold(Expression::from(vec![0.0; theta_len]), |sum, x| sum + x)
            .assign(assignment);

        let value: ConstantValue = result_orig.into();
        let result = value.into_tensor().to_vec();
        result
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use opensrdk_kernel_method::RBF;
    //use opensrdk_linear_algebra::SymmetricPackedMatrix;
    use opensrdk_symbolic_computation::{
        new_partial_variable, new_variable,
        opensrdk_linear_algebra::{Matrix, SymmetricPackedMatrix},
        ConstantValue, Expression, ExpressionArray,
    };
    use rand::{prelude::StdRng, Rng, SeedableRng};
    use rand_distr::{Distribution, StandardNormal};

    use crate::{
        ContinuousDistribution, DistributionProduct, JointDistribution, MultivariateNormal,
    };

    use super::SteinVariationalGradientDescent;
    use opensrdk_kernel_method::*;

    #[test]
    fn it_works() {
        let mut rng = StdRng::from_seed([1; 32]);
        let mut rng2 = StdRng::from_seed([32; 32]);

        let samples_xy = (0..10)
            .into_iter()
            .map(|_| {
                let x = rng2.gen_range(-8.0..=8.0);
                let y = 0.5 * x + rng.sample::<f64, _>(StandardNormal);

                vec![x, y]
            })
            .collect::<Vec<Vec<f64>>>();

        let x = &samples_xy
            .iter()
            .map(|v| vec![1.0, v[0]])
            .collect::<Vec<_>>();
        let y = &samples_xy.iter().map(|v| v[1]).collect::<Vec<_>>();

        let dim = x[0].len();

        let sigma = Expression::from(Matrix::from(1usize, vec![0.1]).unwrap());

        let theta_0 = new_variable("alpha".to_owned());
        let theta_1 = new_variable("beta".to_owned());

        let theta_vec = vec![theta_0.clone(), theta_1.clone()];

        let factory = |i: &[usize]| theta_vec[i[0].clone()].clone();
        let sizes: Vec<usize> = vec![theta_vec.len()];
        let theta_array_orig = ExpressionArray::from_factory(sizes, factory);
        let theta_array = new_partial_variable(theta_array_orig);

        println!("{:?}", theta_array.mathematical_sizes());

        let likelihood = (0..x.len())
            .map(|i| {
                MultivariateNormal::new(
                    Expression::from(y[i]),
                    theta_0.clone() + theta_1.clone() * Expression::from(x[i][0]),
                    sigma.clone(),
                    1usize,
                )
            })
            .distribution_product();

        let prior_sigma = Expression::from(Matrix::from(dim, vec![0.5, 0.5, 0.0, 0.5]).unwrap());
        println!("{:?}", prior_sigma);

        let prior_mu = Expression::from(vec![0.5; dim]);
        let prior = MultivariateNormal::new(theta_array, prior_mu, prior_sigma, dim);

        let kernel = RBF;
        let kernel_params = [0.5];
        let samples_orig = (0..10)
            .into_iter()
            .map(|v| {
                let mut rng3 = StdRng::from_seed([v; 32]);
                let mut rng4 = StdRng::from_seed([v; 32]);
                let theta_0 = rng3.gen_range(-5.0..=5.0);
                let theta_1 = rng4.gen_range(0.0..=10.0) - 5.0;
                vec![theta_0, theta_1]
            })
            .collect::<Vec<_>>();

        let samples = samples_orig
            .iter()
            .map(|samples_orig_elem| {
                let factory = |i: &[usize]| Expression::from(samples_orig_elem[i[0]].clone());
                let sizes: Vec<usize> = vec![2usize];
                let samples_elem = ExpressionArray::from_factory(sizes, factory);
                samples_elem
            })
            .collect::<Vec<ExpressionArray>>();

        println!("{:?}", "one");

        let stein_test = SteinVariationalGradientDescent::new(
            &likelihood,
            &prior,
            &kernel,
            &kernel_params,
            samples.clone(),
        );

        // let hash = HashMap::new();

        let str = &likelihood.condition_ids();
        let str_vec: Vec<_> = str.into_iter().collect();

        let theta_map = &mut HashMap::new();
        let theta_len = str_vec.len();
        let len = &samples.clone()[0].elems().len();

        println!("{:?}", &samples.clone()[0]);
        println!("{:?}", len);
        for i in 0..*len {
            let expression = samples[0].elems().get(&vec![i]).unwrap();

            let elem: ConstantValue = expression.clone().into();

            theta_map.insert(str_vec[i].clone(), elem);
        }
        println!("{:?}", "two");
        println!("{:?}", theta_map);

        let phi = &stein_test.direction(theta_map);
        //let phi = &stein_test.update_sample(theta_map, 3f64);

        println!("{:?}", phi)
    }

    #[test]
    fn it_works5() {
        let kernel_orig = RBF;
        let kernel_params = [0.5];
        let kernel_params_expression = kernel_params
            .iter()
            .map(|elem| Expression::from(*elem))
            .collect::<Vec<Expression>>();

        let samples_orig = (0..10)
            .into_iter()
            .map(|v| {
                let mut rng3 = StdRng::from_seed([v; 32]);
                let mut rng4 = StdRng::from_seed([v; 32]);
                let theta_0 = rng3.gen_range(-5.0..=5.0);
                let theta_1 = rng4.gen_range(0.0..=10.0) - 5.0;
                vec![theta_0, theta_1]
            })
            .collect::<Vec<_>>();

        let samples = samples_orig
            .iter()
            .map(|samples_orig_elem| {
                let factory = |i: &[usize]| Expression::from(samples_orig_elem[i[0]].clone());
                let sizes: Vec<usize> = vec![2usize];
                let samples_elem = ExpressionArray::from_factory(sizes, factory);
                samples_elem
            })
            .collect::<Vec<ExpressionArray>>();

        let samples_array = new_partial_variable(samples[0].clone());

        let theta_vec = vec![
            new_variable("alpha".to_owned()),
            new_variable("beta".to_owned()),
        ];
        let factory = |i: &[usize]| theta_vec[i[0].clone()].clone();
        let sizes: Vec<usize> = vec![theta_vec.len()];
        let theta_array_orig = ExpressionArray::from_factory(sizes, factory);
        let theta_array = new_partial_variable(theta_array_orig);

        let theta_map = &mut HashMap::new();
        theta_map.insert("alpha", ConstantValue::Scalar(3f64));
        theta_map.insert("beta", ConstantValue::Scalar(7f64));

        let kernel = kernel_orig
            .expression(
                theta_array.clone(),
                samples_array.clone(),
                &kernel_params_expression,
            )
            .unwrap();
        let kernel_expression = kernel.clone().assign(theta_map);

        println!("{:#?}", kernel);
        println!("{:#?}", kernel_expression);
    }
}

// #[cfg(test)]
// mod tests {
//     use std::collections::HashMap;

//     use opensrdk_kernel_method::RBF;
//     use opensrdk_symbolic_computation::{
//         new_variable, opensrdk_linear_algebra::Matrix, ConstantValue, Expression, ExpressionArray,
//     };
//     use rand::{prelude::StdRng, Rng, SeedableRng};
//     use rand_distr::{Distribution, StandardNormal};

//     use crate::{
//         ContinuousDistribution, DistributionProduct, JointDistribution, MultivariateNormal,
//     };

//     use super::SteinVariationalGradientDescent;
//     use opensrdk_kernel_method::*;

//     #[test]
//     fn it_works() {
//         let mut rng = StdRng::from_seed([1; 32]);
//         let mut rng2 = StdRng::from_seed([32; 32]);

//         let samples_xy = (0..10)
//             .into_iter()
//             .map(|_| {
//                 let x = rng2.gen_range(-8.0..=8.0);
//                 let y = 0.5 * x + rng.sample::<f64, _>(StandardNormal);

//                 vec![x, y]
//             })
//             .collect::<Vec<Vec<f64>>>();

//         let x = &samples_xy.iter().map(|v| v[0]).collect::<Vec<_>>();
//         let y = &samples_xy.iter().map(|v| v[1]).collect::<Vec<_>>();

//         let sigma = Expression::from(0.5f64);

//         let theta_1 = new_variable("beta".to_owned());

//         let likelihood = (0..x.len())
//             .map(|i| {
//                 MultivariateNormal::new(
//                     Expression::from(y[i]),
//                     theta_1.clone() * Expression::from(x[i]),
//                     sigma.clone(),
//                     1usize,
//                 )
//             })
//             .distribution_product();

//         let dim = 1usize;
//         let prior_sigma =
//             Expression::from(Matrix::from(dim, vec![0.5; dim * (dim + 1) / 2]).unwrap());

//         let prior_mu = Expression::from(vec![0.5; dim]);
//         let prior = MultivariateNormal::new(theta_1, prior_sigma, prior_mu, dim);

//         let kernel = RBF;
//         let kernel_params = [0.5, 0.5];
//         let samples_orig = (0..10)
//             .into_iter()
//             .map(|v| {
//                 let mut rng3 = StdRng::from_seed([v; 32]);
//                 let theta_0 = rng3.gen_range(-5.0..=5.0);
//                 theta_0
//             })
//             .collect::<Vec<f64>>();

//         let samples = samples_orig
//             .iter()
//             .map(|samples_orig_elem| {
//                 let factory = |i: &[usize]| Expression::from(samples_orig_elem.clone());
//                 let sizes: Vec<usize> = vec![1usize];
//                 let samples_elem = ExpressionArray::from_factory(sizes, factory);
//                 samples_elem
//             })
//             .collect::<Vec<ExpressionArray>>();

//         println!("{:?}", "one");

//         let stein_test = SteinVariationalGradientDescent::new(
//             &likelihood,
//             &prior,
//             &kernel,
//             &kernel_params,
//             samples.clone(),
//         );

//         let hash = HashMap::new();

//         // let str = &likelihood.condition_ids();
//         // let str_vec: Vec<_> = str.into_iter().collect();

//         // let theta_map = &mut HashMap::new();
//         // let theta_len = str_vec.len();
//         // let len = &samples.clone()[0].elems().len();

//         // for i in 0..*len {
//         //     let expression = samples[0].elems().get(&vec![i]).unwrap();

//         //     let elem = if let Expression::Constant(value) = expression {
//         //         let constantValue: ConstantValue = value.clone();
//         //         constantValue
//         //     } else {
//         //         panic!("This isn't ConstantValue !");
//         //     };

//         //     theta_map.insert(str_vec[i], elem);
//         // }
//         // println!("{:?}", "two");

//         // let phi = &stein_test.direction(&hash);
//         let phi = &stein_test.update_sample(&hash, 3f64);

//         println!("{:?}", phi)
//     }
// }
