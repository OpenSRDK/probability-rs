// Sampling Importance Resampling
use crate::rand::SeedableRng;
use crate::{Distribution, DistributionError, RandomVariable, SamplesDistribution};
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
        thr: f64,
    ) -> Result<f64, DistributionError> {
        let mut rng = StdRng::from_seed([1; 32]);

        //let mut state = self.state.clone();

        let mut p = (0..self.particles)
            .into_iter()
            .map(|i| -> Result<_, DistributionError> {
                let pi = self.proposal.sample(&self.state, &mut rng)?;
                Ok(pi)
            })
            .collect::<Result<Vec<_>, _>>()?;

        loop {
            let mut w_initial = vec![1.0 / self.particles as f64; self.particles];

            let mut w_orig = (0..self.particles)
                .into_iter()
                .map(|i| -> Result<_, DistributionError> {
                    let wi_orig = w_initial[i]
                        * self.distr_y.fk(&self.value, &p[i])?
                        * self.distr_x.fk(&p[i], &self.state)?
                        / self.proposal.fk(&p[i], &self.state)?;
                    Ok(wi_orig)
                })
                .collect::<Result<Vec<_>, _>>()?;

            let mut w = (0..self.particles)
                .into_iter()
                .map(|i| -> Result<_, DistributionError> {
                    let wi = w_initial[i] / (w_orig.iter().map(|wi_orig| wi_orig).sum::<f64>());
                    Ok(wi)
                })
                .collect::<Result<Vec<_>, _>>()?;

            let mut eff = 1.0 / (w.iter().map(|wi| wi.powi(2)).sum::<f64>());

            if eff > thr {
                break;
            }
            let mut p_sample = vec![];

            for i in 0..w.len() {
                let mut num_w = (1000.0 * w[i]).round() as usize;
                let mut pi_sample = vec![p[i]; num_w];
                let mut p_sample = p_sample.append(&mut pi_sample);
            }

            let mut weighted_distr = SamplesDistribution::new(p_sample);
            let mut p = (0..self.particles)
                .into_iter()
                .map(|i| -> Result<_, DistributionError> {
                    let pi = weighted_distr.sample(&(), &mut rng)?;
                    Ok(pi)
                })
                .collect::<Result<Vec<_>, _>>()?;
        }

        Ok(Y)
    }
}
