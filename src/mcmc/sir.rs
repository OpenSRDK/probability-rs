// Sampling Importance Resampling
use crate::rand::SeedableRng;
use crate::{Distribution, DistributionError, RandomVariable};
use rand::rngs::StdRng;

pub struct ParticleFilter<Y, X, D1, D2, PD>
where
    Y: RandomVariable,
    X: RandomVariable,
    D1: Distribution<Value = Y, Condition = X>,
    D2: Distribution<Value = X, Condition = X>,
    PD: Distribution<Value = X, Condition = X>,
{
    value: Y,
    state: X,
    particles: usize,
    distr_y: D1,
    distr_x: D2,
    proposal: PD,
}

impl<Y, X, D1, D2, PD> ParticleFilter<Y, X, D1, D2, PD>
where
    Y: RandomVariable,
    X: RandomVariable,
    D1: Distribution<Value = Y, Condition = X>,
    D2: Distribution<Value = X, Condition = X>,
    PD: Distribution<Value = X, Condition = X>,
{
    pub fn new(
        value: Y,
        state: X,
        particles: usize,
        distr_y: D1,
        distr_x: D2,
        proposal: PD,
    ) -> Result<Self, DistributionError> {
        Ok(Self {
            value,
            state,
            particles,
            distr_y,
            distr_x,
            proposal,
        })
    }

    pub fn filtering(
        &self,
        f: impl Fn(&X) -> X,
        h: impl Fn(&X) -> Y,
    ) -> Result<f64, DistributionError> {
        let mut rng = StdRng::from_seed([1; 32]);
        let w_initial = vec![1.0 / self.particles as f64; self.particles];
        let x_initial = (0..self.particles)
            .into_iter()
            .map(|i| -> Result<_, DistributionError> {
                let pi = self.proposal.sample(&self.state, &mut rng)?;
                Ok(pi)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let w_orig = w_initial[0]
            * self.distr_y.fk(&self.value, &x_initial[0])?
            * self.distr_x.fk(&x_initial[0], &self.state)?
            / self.proposal.fk(&x_initial[0], &self.state)?;

        Ok(Y)
    }
}
