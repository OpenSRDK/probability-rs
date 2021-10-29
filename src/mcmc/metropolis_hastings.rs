use crate::{Distribution, DistributionError, RandomVariable};
use rand::prelude::*;

/// Sample b from posterior p(b|a) with likelihood p(a|b) and prior p(b)
pub struct MetropolisHastingsSampler<'a, L, P, A, B, PD>
where
    L: Distribution<T = A, U = B>,
    P: Distribution<T = B, U = ()>,
    A: RandomVariable,
    B: RandomVariable,
    PD: Distribution<T = B, U = B>,
{
    value: &'a A,
    likelihood: &'a L,
    prior: &'a P,
    proposal: &'a PD,
}

impl<'a, L, P, A, B, PD> MetropolisHastingsSampler<'a, L, P, A, B, PD>
where
    L: Distribution<T = A, U = B>,
    P: Distribution<T = B, U = ()>,
    A: RandomVariable,
    B: RandomVariable,
    PD: Distribution<T = B, U = B>,
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
        rng: &mut StdRng,
    ) -> Result<B, DistributionError> {
        let mut state = initial;

        for _ in 0..iter {
            let candidate = self.proposal.sample(&state, rng)?;
            let r = (self.likelihood.p(self.value, &candidate)?
                * self.prior.p(&candidate, &())?
                * self.proposal.p(&state, &candidate)?)
                / (self.likelihood.p(self.value, &state)?
                    * self.prior.p(&state, &())?
                    * self.proposal.p(&candidate, &state)?);
            let r = r.min(1.0);
            let p = rng.gen_range(0.0..=1.0);

            if p < r {
                state = candidate;
            }
        }

        Ok(state)
    }
}
