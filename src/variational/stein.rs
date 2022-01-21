use crate::{
    ContinuousSamplesDistribution, Distribution, DistributionError, RandomVariable,
    ValueDifferentiableDistribution,
};
use opensrdk_kernel_method::*;
use opensrdk_linear_algebra::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub struct SteinVariational<'a, L, A, B, K>
where
    L: Distribution<Value = A, Condition = B> + ValueDifferentiableDistribution,
    A: RandomVariable,
    B: RandomVariable,
    K: PositiveDefiniteKernel<Vec<f64>> + ValueDifferentiable<Vec<f64>>,
{
    value: &'a A,
    likelihood: &'a L,
    kernel: &'a K,
    kernel_params: &'a [f64],
    samples: &'a mut ContinuousSamplesDistribution<Vec<f64>>,
}

impl<'a, L, A, B, K> SteinVariational<'a, L, A, B, K>
where
    L: Distribution<Value = A, Condition = B> + ValueDifferentiableDistribution,
    A: RandomVariable,
    B: RandomVariable,
    K: PositiveDefiniteKernel<Vec<f64>> + ValueDifferentiable<Vec<f64>>,
{
    pub fn new(
        value: &'a A,
        likelihood: &'a L,
        kernel: &'a K,
        kernel_params: &'a [f64],
        samples: &'a mut ContinuousSamplesDistribution<Vec<f64>>,
    ) -> Self {
        Self {
            value,
            likelihood,
            kernel,
            kernel_params,
            samples,
        }
    }

    pub fn direction(&self, x: &Vec<f64>, theta: &B) -> Result<Vec<f64>, DistributionError> {
        let n = self.samples.samples().len();
        let phi = (0..n)
            .into_par_iter()
            .map(|j| &self.samples.samples()[j])
            .map(|xj| {
                let kernel = self.kernel.value(self.kernel_params, &x, &xj).unwrap();
                let kernel_diff = self
                    .kernel
                    .ln_diff_value(self.kernel_params, &x, &xj)
                    .unwrap()
                    .0
                    .col_mat();
                let l_diff = self
                    .likelihood
                    .ln_diff_value(self.value, theta)
                    .unwrap()
                    .col_mat();
                let p_diff = l_diff;
                kernel * p_diff + kernel_diff
            })
            .reduce(|| mat!(0.0), |sum, x| sum + x);

        Ok(phi.vec())
    }
}
