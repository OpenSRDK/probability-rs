use crate::{Distribution, DistributionError, RandomVariable, SamplableDistribution};
use rand::prelude::*;

/// Sample b from posterior p(b|a) with likelihood p(a|b) and prior p(b)
pub struct MetropolisHastingsSampler<'a, L, P, A, B, PD>
where
    L: Distribution<Value = A, Condition = B>,
    P: Distribution<Value = B, Condition = ()>,
    A: RandomVariable,
    B: RandomVariable,
    PD: SamplableDistribution<Value = B, Condition = B>,
{
    value: &'a A,
    likelihood: &'a L,
    prior: &'a P,
    proposal: &'a PD,
}

impl<'a, L, P, A, B, PD> MetropolisHastingsSampler<'a, L, P, A, B, PD>
where
    L: Distribution<Value = A, Condition = B>,
    P: Distribution<Value = B, Condition = ()>,
    A: RandomVariable,
    B: RandomVariable,
    PD: SamplableDistribution<Value = B, Condition = B>,
{
    pub fn new(value: &'a A, likelihood: &'a L, prior: &'a P, proposal: &'a PD) -> Self {
        Self {
            value,
            likelihood,
            prior,
            proposal,
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
            let r = (self.likelihood.p_kernel(self.value, &candidate)?
                * self.prior.p_kernel(&candidate, &())?
                * self.proposal.p_kernel(&state, &candidate)?)
                / (self.likelihood.p_kernel(self.value, &state)?
                    * self.prior.p_kernel(&state, &())?
                    * self.proposal.p_kernel(&candidate, &state)?);
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
