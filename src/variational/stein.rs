use crate::{
    ConditionDifferentiableDistribution, ContinuousSamplesDistribution, Distribution,
    DistributionError, RandomVariable, ValueDifferentiableDistribution,
};
use opensrdk_kernel_method::*;
use opensrdk_linear_algebra::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Adjust samples {b} from posterior p(b|a) with likelihood p(a|b) and prior p(b)
pub struct SteinVariational<'a, L, P, A, B, K>
where
    L: Distribution<Value = A, Condition = B> + ConditionDifferentiableDistribution,
    P: Distribution<Value = B, Condition = ()> + ValueDifferentiableDistribution,
    A: RandomVariable,
    B: RandomVariable,
    K: PositiveDefiniteKernel<Vec<f64>>,
{
    value: &'a A,
    likelihood: &'a L,
    prior: &'a P,
    kernel: &'a K,
    kernel_params: &'a [f64],
    samples: &'a mut ContinuousSamplesDistribution<Vec<f64>>,
}

impl<'a, L, P, A, B, K> SteinVariational<'a, L, P, A, B, K>
where
    L: Distribution<Value = A, Condition = B>,
    P: Distribution<Value = B, Condition = ()>,
    A: RandomVariable,
    B: RandomVariable,
    K: PositiveDefiniteKernel<Vec<f64>>,
{
    pub fn new(
        value: &'a A,
        likelihood: &'a L,
        prior: &'a P,
        kernel: &'a K,
        kernel_params: &'a [f64],
        samples: &'a mut ContinuousSamplesDistribution<Vec<f64>>,
    ) -> Self {
        Self {
            value,
            likelihood,
            prior,
            kernel,
            kernel_params,
            samples,
        }
    }

    pub fn samples(&mut self) -> &mut ContinuousSamplesDistribution<Vec<f64>> {
        &mut self.samples
    }

    pub fn direction(&self, theta: &B) -> Result<Vec<f64>, DistributionError> {
        let n = self.samples.samples().len();
        let m = self.samples.samples()[0].len();
        let theta_vec = theta.clone().transform_vec().0;
        let phi_sum = self
            .samples
            .samples()
            .iter()
            .map(|theta_j| {
                let kernel = self
                    .kernel
                    .value(self.kernel_params, &theta_vec, &theta_j)
                    .unwrap();
                let kernel_diff = self
                    .kernel
                    .ln_diff_value(self.kernel_params, &theta_vec, &theta_j)
                    .unwrap()
                    .col_mat();
                let p_diff = self
                    .likelihood
                    .ln_diff_condition(self.value, &theta)
                    .unwrap()
                    .col_mat()
                    + self.prior.ln_diff_value(&theta, &()).unwrap().col_mat();
                kernel * p_diff + kernel_diff
            })
            .fold(vec![0.0; m].col_mat(), |sum, x| sum + x);
        let phi = phi_sum
            .vec()
            .iter()
            .map(|i| i / n as f64)
            .collect::<Vec<f64>>();
        Ok(phi)
    }
}

// #[cfg(test)]
// mod tests {
//     use opensrdk_kernel_method::RBF;
//     use opensrdk_linear_algebra::{mat, SymmetricPackedMatrix};
//     use rand::{prelude::StdRng, Rng, SeedableRng};
//     use rand_distr::StandardNormal;

//     use crate::{
//         ConditionMappableDistribution, ContinuousSamplesDistribution,
//         DifferentiableConditionMappedDistribution, DistributionValueProduct, ExactEllipticalParams,
//         MultivariateNormal, Normal, NormalParams,
//     };

//     use super::SteinVariational;
//     use opensrdk_kernel_method::*;

//     #[test]
//     fn it_works() {
//         let mut rng = StdRng::from_seed([1; 32]);
//         let mut rng2 = StdRng::from_seed([32; 32]);
//         let samples_xy = (0..20)
//             .into_iter()
//             .map(|_| {
//                 let x = rng2.gen_range(-8.0..=8.0);
//                 let y = 1.0 + 0.5 * x + rng.sample::<f64, _>(StandardNormal);

//                 vec![x, y]
//             })
//             .collect::<Vec<Vec<f64>>>();

//         let x = &samples_xy
//             .iter()
//             .map(|v| vec![1.0, v[0]])
//             .collect::<Vec<_>>();
//         let y = &samples_xy.iter().map(|v| v[1]).collect::<Vec<_>>();
//         let sigma = 0.5;

//         let value = y.clone();

//         let likelihood = x
//             .into_iter()
//             .map(|xi| {
//                 let likelihood_i = Normal.map_condition(move |theta: &Vec<f64>| {
//                     NormalParams::new(theta[0] * xi[0] + theta[1] * xi[1], sigma)
//                 });
//                 let condition_diff = move |_theta: &Vec<f64>| {
//                     mat!(xi[0],0.0;
//                       xi[1], 0.0)
//                 };
//                 let likelihood_i_diff =
//                     DifferentiableConditionMappedDistribution::new(likelihood_i, condition_diff);
//                 likelihood_i_diff
//             })
//             .only_value_joint();

//         let dim = x[0].len();
//         let prior_sigma_sym =
//             SymmetricPackedMatrix::from(dim, vec![0.5; dim * (dim + 1) / 2]).unwrap();
//         let prior_sigma = prior_sigma_sym.pptrf().unwrap();

//         let prior_mu = vec![0.5; dim];
//         let prior_params =
//             ExactEllipticalParams::new(prior_mu.clone(), prior_sigma.clone()).unwrap();
//         let prior = MultivariateNormal::new().map_condition(|_| Ok(prior_params.clone()));

//         let kernel = RBF;
//         let kernel_params = [0.5, 0.5];
//         let samples_orig = (0..10)
//             .into_iter()
//             .map(|v| {
//                 let mut rng3 = StdRng::from_seed([v; 32]);
//                 let theta_0 = rng3.gen_range(-5.0..=5.0);
//                 let mut rng4 = StdRng::from_seed([v * 2; 32]);
//                 let theta_1 = rng4.gen_range(-5.0..=5.0);
//                 vec![theta_0, theta_1]
//             })
//             .collect::<Vec<Vec<f64>>>();
//         let samples = &mut ContinuousSamplesDistribution::new(samples_orig);

//         let theta = vec![0.1, 0.1];

//         let stein_test = SteinVariational::new(
//             &value,
//             &likelihood,
//             &prior,
//             &kernel,
//             &kernel_params,
//             samples,
//         );

//         let stein_ref = &stein_test;
//         let phi = stein_ref.direction(&theta).unwrap();

//         println!("{:?}", phi)
//     }
// }
