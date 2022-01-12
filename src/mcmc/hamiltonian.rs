use crate::{Distribution, DistributionError, RandomVariable};
use rand::prelude::*;

/// Sample b from posterior p(b|a) with likelihood p(a|b) and prior p(b)
pub struct HamiltonianSampler<'a, L, P, A, B>
where
    L: Distribution<Value = A, Condition = B>,
    P: Distribution<Value = B, Condition = ()>,
    A: RandomVariable,
    B: RandomVariable,
{
    value: &'a A,
    likelihood: &'a L,
    prior: &'a P,
}

impl<'a, L, P, A, B> HamiltonianSampler<'a, L, P, A, B>
where
    L: Distribution<Value = A, Condition = B>,
    P: Distribution<Value = B, Condition = ()>,
    A: RandomVariable,
    B: RandomVariable,
{
    pub fn new(value: &'a A, likelihood: &'a L, prior: &'a P) -> Self {
        Self {
            value,
            likelihood,
            prior,
        }
    }

    pub fn sample(
        &self,
        x: f64,
        max_iter: usize,
        rng: &mut dyn RngCore,
    ) -> Result<f64, DistributionError> {
        let mut st = 0.0;
        let mut ed = 1.0;

        let r = shrink(x)?;
        let slice = self.likelihood.fk(&self.value, &x)? * self.prior.fk(&x, &())?
            - 2.0 * rng.gen_range(0.0f64..1.0f64).ln();

        for _iter in 0..max_iter {
            let rnew = rng.gen_range(st..ed);
            let expanded = expand(rnew)?;

            let newlik = self.likelihood.fk(&self.value, &expanded)?
                * self.prior.fk(&expanded, &())?
                - (2.0 * rnew * (1.0 - rnew));

            if newlik > slice {
                return expand(rnew);
            } else if rnew > r {
                ed = rnew;
            } else if rnew < r {
                st = rnew;
            } else {
                return Ok(x);
            }
        }
        Ok(x)
    }
}

fn hamiltonian(p: f64, theta: f64) -> Result<f64, DistributionError> {
    let lambda = 1.0;
    let alpha = 1.0;
    Ok(lambda * theta - (alpha - 1.0) * theta.ln() + 0.5 * p.powf(2.0))
}

fn leapfrog_next_half_p(p: f64, theta: f64) -> Result<f64, DistributionError> {
    let lambda = 1.0;
    let alpha = 1.0;
    let eps = 0.01;
    Ok(p - 0.5 * eps * (lambda - (alpha - 1.0) / theta))
}

fn leapfrog_next_theta(p: f64, theta: f64) -> Result<f64, DistributionError> {
    let eps = 0.01;
    Ok(theta + eps * p)
}

fn move_one_step(p: f64, theta: f64) -> Result<Vec<f64>, DistributionError> {
    let L = 100;
    let p_sample = vec![];
    p_sample.append(&mut vec![1.0, p, theta, hamiltonian(p, theta)?]);
    for _i in 0..L {
        p = leapfrog_next_half_p(p, theta)?;
        theta = leapfrog_next_theta(p, theta)?;
        p = leapfrog_next_half_p(p, theta)?;
        p_sample.append(&mut vec![1.0, p, theta, hamiltonian(p, theta)?])
    }
    Ok(p_sample)
}

// r=exp⁡(H(θ(t),p(t))−H(θ(a),p(a)))
// これと標準正規分布を比較して受容、非受容を決定
