use crate::{Distribution, DistributionError, RandomVariable};

/// Sample b from posterior p(b|a) with likelihood p(a|b) and prior p(b)

pub struct ImportanceSampler<D, A, B, PD>
where
    D: Distribution<T = A, U = B>,
    A: RandomVariable,
    B: RandomVariable,
    PD: Distribution<T = B, U = ()>,
{
    value: A,
    distribution: D
    proposal: PD,
}

#[derive(thiserror::Error, Debug)]
pub enum ImportanceSamplingError {
    #[error("out of range")]
    OutOfRange,
    #[error("Unknown error")]
    Unknown,
}

impl<D, A, B, PD> ImportanceSampler<D, A, B, PD>
where
    D: Distribution<T = B, U = ()>,
    A: RandomVariable,
    B: RandomVariable,
    PD: Distribution<T = B, U = ()>,
{
    pub fn new(value: A, distribution: D, proposal: PD) -> Result<Self, DistributionError> {
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
                let wi = self.distribution.fk(&xi, &())?
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
