// Sampling Importance Resampling
use crate::rand::SeedableRng;
use crate::VectorSampleable;
use crate::{ContinuousSamplesDistribution, Distribution, DistributionError, RandomVariable};
use rand::rngs::StdRng;
use std::hash::Hash;
use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::Div;

pub struct ParticleFilter<Y, X, D1, D2, PD>
where
    Y: RandomVariable,
    X: RandomVariable + Eq + Hash + VectorSampleable + Sum + Div<f64, Output = X>,
    D1: Distribution<Value = Y, Condition = X>,
    D2: Distribution<Value = X, Condition = X>,
    PD: Distribution<Value = X, Condition = X>,
{
    observable: Vec<Y>,
    distr_y: D1,
    distr_x: D2,
    proposal: PD,
    phantom: PhantomData<X>,
}

impl<Y, X, D1, D2, PD> ParticleFilter<Y, X, D1, D2, PD>
where
    Y: RandomVariable,
    X: RandomVariable + Eq + Hash + VectorSampleable + Sum + Div<f64, Output = X>,
    D1: Distribution<Value = Y, Condition = X>,
    D2: Distribution<Value = X, Condition = X>,
    PD: Distribution<Value = X, Condition = X>,
{
    pub fn new(
        observable: Vec<Y>,
        distr_y: D1,
        distr_x: D2,
        proposal: PD,
    ) -> Result<Self, DistributionError> {
        Ok(Self {
            observable,
            distr_y,
            distr_x,
            proposal,
            phantom: PhantomData,
        })
    }

    pub fn filtering(
        &self,
        particles_initial: Vec<X>,
        thr: f64,
    ) -> Result<Vec<ContinuousSamplesDistribution<X>>, DistributionError> {
        let mut rng = StdRng::from_seed([1; 32]);

        let mut distr_vec = vec![];

        let particles_len = particles_initial.len();

        let mut p_previous = particles_initial;

        let w_initial = vec![1.0 / particles_len as f64; particles_len];

        let mut w_previous = w_initial;

        for t in 0..(self.observable).len() {
            let mut p = (0..particles_len)
                .into_iter()
                .map(|i| -> Result<_, DistributionError> {
                    let pi = self.proposal.sample(&p_previous[i], &mut rng)?;
                    Ok(pi)
                })
                .collect::<Result<Vec<_>, _>>()?;

            loop {
                let w_orig = (0..particles_len)
                    .into_iter()
                    .map(|i| -> Result<_, DistributionError> {
                        let wi_orig = w_previous[i]
                            * self.distr_y.fk(&self.observable[t], &p[i])?
                            * self.distr_x.fk(&p[i], &p_previous[i])?
                            / self.proposal.fk(&p[i], &p_previous[i])?;
                        Ok(wi_orig)
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let w = (0..particles_len)
                    .into_iter()
                    .map(|i| -> Result<_, DistributionError> {
                        let wi =
                            w_previous[i] / (w_orig.iter().map(|wi_orig| wi_orig).sum::<f64>());
                        Ok(wi)
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let eff = 1.0 / (w.iter().map(|wi| wi.powi(2)).sum::<f64>());

                let mut p_sample = vec![];

                for i in 0..w.len() {
                    let num_w = (particles_len as f64 * 100.0 * w[i]).round() as usize;
                    let mut pi_sample = vec![p[i].clone(); num_w];
                    p_sample.append(&mut pi_sample);
                }

                let weighted_distr = ContinuousSamplesDistribution::new(p_sample);

                let mut weighted_distr_vec = vec![weighted_distr.clone()];

                if eff > thr {
                    distr_vec.append(&mut weighted_distr_vec);
                    p_previous = p;
                    w_previous = w;
                    break;
                }

                p = (0..particles_len)
                    .into_iter()
                    .map(|_i| -> Result<_, DistributionError> {
                        let pi = weighted_distr.sample(&(), &mut rng)?;
                        Ok(pi)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            }
        }
        Ok(distr_vec)
    }
}
