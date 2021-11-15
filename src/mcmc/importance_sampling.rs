use crate::{Distribution, DistributionError, RandomVariable};

/// Sample b from posterior p(b|a) with likelihood p(a|b) and prior p(b)

pub struct ImportanceSampler<T, D, PD>
where
    T: RandomVariable,
    D: Distribution<T = T, U = ()>,
    PD: Distribution<T = T, U = ()>,
{
    distribution: D,
    proposal: PD,
}

#[derive(thiserror::Error, Debug)]
pub enum ImportanceSamplingError {
    #[error("out of range")]
    OutOfRange,
    #[error("Unknown error")]
    Unknown,
}

impl<T, D, PD> ImportanceSampler<T, D, PD>
where
    T: RandomVariable,
    D: Distribution<T = T, U = ()>,
    PD: Distribution<T = T, U = ()>,
{
    pub fn new(distribution: D, proposal: PD) -> Result<Self, DistributionError> {
        Ok(Self {
            distribution,
            proposal,
        })
    }

    pub fn expectation(&self, f: impl Fn(&T) -> f64, x: &[T]) -> Result<f64, DistributionError> {
        let wi_fxi = x
            .iter()
            .map(|xi| -> Result<_, DistributionError> {
                let wi = self.distribution.fk(&xi, &())? / self.proposal.fk(&xi, &())?;
                let fxi = f(xi);
                Ok((wi, fxi))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let sum_w = wi_fxi.iter().map(|&(wi, _)| wi).sum::<f64>();
        let sum_w_fx = wi_fxi.iter().map(|&(wi, fxi)| wi * fxi).sum::<f64>();

        Ok(sum_w_fx / sum_w)
    }
}
