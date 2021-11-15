// Sampling Importance Resampling
use crate::{Distribution, DistributionError, RandomVariable};

pub struct ParticleFilter<Y, X, PD>
where
    Y: RandomVariable,
    X: RandomVariable,
    PD: Distribution<T = X, U = ()>,
{
    value: Y,
    state: X,
    particles: usize,
    proposal: PD,
}

impl<Y, X, PD> ParticleFilter<Y, X, PD>
where
    Y: RandomVariable,
    X: RandomVariable,
    PD: Distribution<T = X, U = ()>,
{
    pub fn new(
        value: Y,
        state: X,
        particles: usize,
        proposal: PD,
    ) -> Result<Self, DistributionError> {
        Ok(Self {
            value,
            state,
            particles,
            proposal,
        })
    }

    pub fn filtering(
        &self,
        f: impl Fn(&X) -> X,
        h: impl Fn(&X) -> Y,
    ) -> Result<f64, DistributionError> {
        let wi_fxi = x
            .iter()
            .map(|xi| -> Result<_, DistributionError> {
                let wl = self.distribution.fk(&xi, &())? / self.proposal.fk(&xi, &())?;
                let fxi = f(xi);
                Ok((wi, fxi))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Y)
    }
}
