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
        iter: usize,
        initial: B,
        rng: &mut dyn RngCore,
    ) -> Result<B, DistributionError> {
        let mut state = initial;
        let mut count = 0;

        while count < iter {
            let candidate = self.proposal.sample(&state, rng)?;
            let r = (self.likelihood.fk(self.value, &candidate)?
                * self.prior.fk(&candidate, &())?
                * self.proposal.fk(&state, &candidate)?)
                / (self.likelihood.fk(self.value, &state)?
                    * self.prior.fk(&state, &())?
                    * self.proposal.fk(&candidate, &state)?);
            let r = r.min(1.0);
            let p = rng.gen_range(0.0..=1.0);

            if p < r {
                state = candidate;
                count += 1;
            }
        }

        Ok(state)
    }
}

fn hamiltonian(p: f64, theta: f64) -> Result<f64, DistributionError> {
    let lambda = 1.0;
    let alpha = 1.0;
    Ok(lambda * theta - (alpha - 1.0) * theta.ln() + 0.5 * p.powf(2.0))
}

fn leapfrog_next_half(p: f64, theta: f64) -> Result<f64, DistributionError> {
    let lambda = 1.0;
    let alpha = 1.0;
    Ok(p - 0.5 * 0.01 * (lambda - (alpha - 1.0) / theta))
}

fn leapfrog_next(p: f64, theta: f64) -> Result<f64, DistributionError> {
    Ok(theta + 0.01 * p)
}

// fn move_one_step(p: f64, theta: f64) -> Result<Vec<X>, DistributionError> {
//     let L = 100.0;
//     let p_sample = vec![];
//     p_sample.append(1, p, theta, hamiltonian(p, theta));
// }
