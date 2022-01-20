use crate::{ContinuousSamplesDistribution, Distribution, DistributionError, RandomVariable};
use opensrdk_kernel_method::*;
use opensrdk_linear_algebra::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub struct SteinVariational<'a, L, P, A, B, K>
where
    L: Distribution<Value = A, Condition = B>,
    P: Distribution<Value = B, Condition = ()>,
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
    L: Distribution<Value = A, Condition = B>,
    P: Distribution<Value = B, Condition = ()>,
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

    pub fn direction(&self, x: Vec<f64>) -> Result<f64, DistributionError> {
        let n = self.samples.samples().len();
        let phi = (0..n)
            .into_par_iter()
            .map(|j| &self.samples.samples()[j])
            .map(|xj| self.kernel.value(self.kernel_params, &x, &xj).unwrap())
            .sum::<f64>();
        let ln_kernel = (0..n)
            .into_par_iter()
            .map(|j| &self.samples.samples()[j])
            .map(|xj| {
                self.kernel
                    .ln_diff_value(self.kernel_params, &x, &xj)
                    .unwrap()
                    .0
                    .col_mat()
            })
            .reduce(|| mat!(0.0), |sum, x| sum + x);
        // let ln_probability = (0..n)
        //     .into_par_iter()
        //     .map(|j| &self.samples.samples()[j])
        //     .map(|xj| {
        //         Ok(self.likelihood.fk(&x, theta)? * self.prior.fk(&x, theta)?)
        //             .ln_diff_value(self.kernel_params, &x, &xj)
        //             .unwrap()
        //             .0
        //             .col_mat()
        //     })
        //     .reduce(|| mat!(0.0), |sum, x| sum + x);
        Ok(phi)
    }
}
