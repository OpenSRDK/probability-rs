// Sampling Importance Resampling
use crate::elliptical::*;
use crate::rand::SeedableRng;
use crate::{Distribution, DistributionError, RandomVariable, SamplesDistribution};
use rand::rngs::StdRng;
use std::hash::Hash;
use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::Div;
pub struct ParticleFilter<Y, X, D1, D2, PD>
where
    Y: RandomVariable,
    X: RandomVariable + Eq + Hash + Sum + Div,
    D1: Distribution<Value = Y, Condition = X>,
    D2: Distribution<Value = X, Condition = X>,
    PD: Distribution<Value = X, Condition = X>,
{
    value: Y,
    distr_y: D1,
    distr_x: D2,
    proposal: PD,
    phantom: PhantomData<X>,
}

impl<Y, X, D1, D2, PD> ParticleFilter<Y, X, D1, D2, PD>
where
    Y: RandomVariable,
    X: RandomVariable + Eq + Hash + Sum + Div,
    D1: Distribution<Value = Y, Condition = X>,
    D2: Distribution<Value = X, Condition = X>,
    PD: Distribution<Value = X, Condition = X>,
{
    pub fn new(
        value: Y,
        distr_y: D1,
        distr_x: D2,
        proposal: PD,
    ) -> Result<Self, DistributionError> {
        Ok(Self {
            value,
            distr_y,
            distr_x,
            proposal,
            phantom: PhantomData,
        })
    }

    pub fn filtering(
        &self,
        particles: usize,
        thr: f64,
    ) -> Result<SamplesDistribution<X>, DistributionError> {
        let mut rng = StdRng::from_seed([1; 32]);

        let x_initial = Normal.sample(&NormalParams::new(0.0, 1.0).unwrap(), &mut rng)?;
        //mu はy[0]を取るのが良いのではないか。

        let mut x = x_initial;

        //for l in 0..y.len()+1{};

        let mut p = (0..particles)
            .into_iter()
            .map(|_i| -> Result<_, DistributionError> {
                let pi = self.proposal.sample(&x, &mut rng)?;
                Ok(pi)
            })
            .collect::<Result<Vec<_>, _>>()?;

        loop {
            let w_initial = vec![1.0 / particles as f64; particles];

            let w_orig = (0..particles)
                .into_iter()
                .map(|i| -> Result<_, DistributionError> {
                    let wi_orig = w_initial[i]
                        * self.distr_y.fk(&self.value, &p[i])?
                        * self.distr_x.fk(&p[i], &self.state)?
                        / self.proposal.fk(&p[i], &self.state)?;
                    Ok(wi_orig)
                })
                .collect::<Result<Vec<_>, _>>()?;

            let w = (0..particles)
                .into_iter()
                .map(|i| -> Result<_, DistributionError> {
                    let wi = w_initial[i] / (w_orig.iter().map(|wi_orig| wi_orig).sum::<f64>());
                    Ok(wi)
                })
                .collect::<Result<Vec<_>, _>>()?;

            let eff = 1.0 / (w.iter().map(|wi| wi.powi(2)).sum::<f64>());

            if eff > thr {
                break;
            }

            let mut p_sample = vec![];

            for i in 0..w.len() {
                let num_w = (particles as f64 * 10.0 * w[i]).round() as usize;
                let mut pi_sample = vec![p[i].clone(); num_w];
                p_sample.append(&mut pi_sample);
            }

            x = p_sample.iter().mean();

            let weighted_distr = SamplesDistribution::new(p_sample);

            p = (0..particles)
                .into_iter()
                .map(|_i| -> Result<_, DistributionError> {
                    let pi = weighted_distr.sample(&(), &mut rng)?;
                    Ok(pi)
                })
                .collect::<Result<Vec<_>, _>>()?;
        }

        let x_distr = SamplesDistribution::new(p);

        Ok(x_distr)
    }
}
