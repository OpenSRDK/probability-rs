use crate::{ConditionDifferentiableDistribution, Distribution, DistributionError, RandomVariable};
use opensrdk_linear_algebra::Vector;

/// Inference q(b|z) which approximates posterior p(b|a) with likelihood p(a|b) and prior p(b)
pub struct VariationalInference<'a, Q, L, P, A, B>
where
    Q: Distribution<Value = A, Condition = Vec<f64>> + ConditionDifferentiableDistribution,
    L: Distribution<Value = A, Condition = B>,
    P: Distribution<Value = B, Condition = ()>,
    A: RandomVariable,
    B: RandomVariable,
{
    approximating_posterior: &'a Q,
    value: &'a A,
    likelihood: &'a L,
    prior: &'a P,
}

impl<'a, Q, L, P, A, B> VariationalInference<'a, Q, L, P, A, B>
where
    Q: Distribution<Value = A, Condition = Vec<f64>> + ConditionDifferentiableDistribution,
    L: Distribution<Value = A, Condition = B>,
    P: Distribution<Value = B, Condition = ()>,
    A: RandomVariable,
    B: RandomVariable,
{
    pub fn new(
        approximating_posterior: &'a Q,
        value: &'a A,
        likelihood: &'a L,
        prior: &'a P,
    ) -> Self {
        Self {
            approximating_posterior,
            value,
            likelihood,
            prior,
        }
    }

    pub fn dkl_dz_for_optimization(
        &self,
        z: &Vec<f64>,
        theta: &B,
    ) -> Result<Vec<f64>, DistributionError> {
        let dq_dz = self
            .approximating_posterior
            .ln_diff_condition(self.value, z)?;
        let q = self.approximating_posterior.fk(self.value, z)?;
        let p = self.likelihood.fk(self.value, theta)? * self.prior.fk(theta, &())?;
        // q.ln() - p.ln() + 1.0 = 0.0
        // minimize (q.ln() - p.ln() + 1.0).powf(2.0) with z?
        Ok(((2.0 * q.ln() + 2.0 * (1.0 - p.ln())) / q * dq_dz.col_mat()).vec())
    }
}
