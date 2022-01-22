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
    K: PositiveDefiniteKernel<Vec<f64>> + ValueDifferentiable<Vec<f64>>,
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
    L: Distribution<Value = A, Condition = B> + ConditionDifferentiableDistribution,
    P: Distribution<Value = B, Condition = ()> + ValueDifferentiableDistribution,
    A: RandomVariable,
    B: RandomVariable,
    K: PositiveDefiniteKernel<Vec<f64>> + ValueDifferentiable<Vec<f64>>,
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

    pub fn direction(&self, theta: &B) -> Result<Vec<f64>, DistributionError> {
        let n = self.samples.samples().len();
        let theta_vec = theta.clone().transform_vec().0;
        let phi = (0..n)
            .into_par_iter()
            .map(|j| &self.samples.samples()[j])
            .map(|theta_j| {
                let kernel = self
                    .kernel
                    .value(self.kernel_params, &theta_vec, &theta_j)
                    .unwrap();
                let kernel_diff = self
                    .kernel
                    .ln_diff_value(self.kernel_params, &theta_vec, &theta_j)
                    .unwrap()
                    .0
                    .col_mat();
                let p_diff = self
                    .likelihood
                    .ln_diff_condition(self.value, &theta)
                    .unwrap()
                    .col_mat()
                    + self.prior.ln_diff_value(&theta, &()).unwrap().col_mat();
                kernel * p_diff + kernel_diff
            })
            .reduce(
                || vec![0.0; theta_vec.len()].col_mat(),
                |sum, theta| sum + theta,
            );

        Ok(phi.vec())
    }
}
