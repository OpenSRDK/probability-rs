use crate::{Distribution, DistributionError, RandomVariable};

/// Sample b from posterior p(b|a) with likelihood p(a|b) and prior p(b)

pub struct ImportanceSampler<L, P, A, B, PD>
where
    L: Distribution<T = A, U = B>,
    P: Distribution<T = B, U = ()>,
    A: RandomVariable,
    B: RandomVariable,
    PD: Distribution<T = B, U = ()>,
{
    value: A,
    likelihood: L,
    prior: P,
    proposal: PD,
}

#[derive(thiserror::Error, Debug)]
pub enum ImportanceSamplingError {
    #[error("out of range")]
    OutOfRange,
    #[error("Unknown error")]
    Unknown,
}

impl<L, P, A, B, PD> ImportanceSampler<L, P, A, B, PD>
where
    L: Distribution<T = A, U = B>,
    P: Distribution<T = B, U = ()>,
    A: RandomVariable,
    B: RandomVariable,
    PD: Distribution<T = B, U = ()>,
{
    pub fn new(value: A, likelihood: L, prior: P, proposal: PD) -> Result<Self, DistributionError> {
        Ok(Self {
            value,
            likelihood,
            prior,
            proposal,
        })
    }

    pub fn expectation(&self, f: impl Fn(&B) -> f64, x: &[B]) -> Result<f64, DistributionError> {
        let wi_fxi = x
            .iter()
            .map(|xi| -> Result<_, DistributionError> {
                let wi = self.likelihood.fk(&self.value, &xi)? * self.prior.fk(&xi, &())?
                    / self.proposal.fk(&xi, &())?;
                let fxi = f(xi);
                Ok((wi, fxi))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let sum_w = wi_fxi.iter().map(|&(wi, _)| wi).sum::<f64>();
        let sum_w_fx = wi_fxi.iter().map(|&(wi, fxi)| wi * fxi).sum::<f64>();

        Ok(sum_w_fx / sum_w)
    }
}
