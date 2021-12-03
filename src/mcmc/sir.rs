// Sampling Importance Resampling
use crate::rand::SeedableRng;
use crate::VectorSampleable;
use crate::{ContinuousSamplesDistribution, Distribution, DistributionError, RandomVariable};
use rand::rngs::StdRng;
use std::hash::Hash;
use std::iter::Sum;
use std::ops::Div;

pub struct ParticleFilter<Y, X, DY, DX, PD>
where
    Y: RandomVariable,
    X: RandomVariable + Eq + Hash + VectorSampleable + Sum + Div<f64, Output = X>,
    DY: Distribution<Value = Y, Condition = X>,
    DX: Distribution<Value = X, Condition = X>,
    PD: Distribution<Value = X, Condition = (Vec<X>, Vec<Y>)>,
{
    observable: Vec<Y>,
    distr_x: DX,
    distr_y: DY,
    proposal: PD,
}

impl<Y, X, DY, DX, PD> ParticleFilter<Y, X, DY, DX, PD>
where
    Y: RandomVariable,
    X: RandomVariable + Eq + Hash + VectorSampleable + Sum + Div<f64, Output = X>,
    DY: Distribution<Value = Y, Condition = X>,
    DX: Distribution<Value = X, Condition = X>,
    PD: Distribution<Value = X, Condition = (Vec<X>, Vec<Y>)>,
{
    pub fn new(
        observable: Vec<Y>,
        distr_x: DX,
        distr_y: DY,
        proposal: PD,
    ) -> Result<Self, DistributionError> {
        Ok(Self {
            observable,
            distr_y,
            distr_x,
            proposal,
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

        let mut p_previous = particles_initial.clone();

        let w_initial = vec![1.0 / particles_len as f64; particles_len];

        let mut w_previous = w_initial;

        let mut vecvec_p = (0..particles_len)
            .into_iter()
            .map(|i| -> Result<_, DistributionError> {
                let vec_pi = vec![particles_initial[i].clone()];
                Ok(vec_pi)
            })
            .collect::<Result<Vec<_>, _>>()?;

        for t in 0..(self.observable).len() {
            let mut p = (0..particles_len)
                .into_iter()
                .map(|i| -> Result<_, DistributionError> {
                    let pi = self.proposal.sample(
                        &(
                            (vecvec_p[i][0..t]).to_vec(),
                            (self.observable[0..t]).to_vec(),
                        ),
                        &mut rng,
                    )?;
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
                            / self.proposal.fk(
                                &p[i],
                                &(
                                    (vecvec_p[i][0..t]).to_vec(),
                                    (self.observable[0..t]).to_vec(),
                                ),
                            )?;
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

                vecvec_p = (0..particles_len)
                    .into_iter()
                    .map(|i| -> Result<_, DistributionError> {
                        let mut vec_pi = vec![p[i].clone()];
                        vecvec_p[i].to_vec().append(&mut vec_pi);
                        let vecvec_pi = vecvec_p[i].clone();
                        Ok(vecvec_pi)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            }
        }
        Ok(distr_vec)
    }
}

#[cfg(test)]
mod tests {
    use crate::distribution::Distribution;
    use crate::ConditionedDistribution;
    use crate::*;
    use opensrdk_linear_algebra::mat;
    use rand::prelude::*;
    use std::hash::Hash;

    #[test]
    fn it_works() {
        // create test data
        let x_sigma = 1.0;
        let y_sigma = 2.0;
        let mut rng = StdRng::from_seed([1; 32]);
        let mut x_series = vec![];
        let mut x_pre = 0.0;
        let mut y_series = vec![];
        for _i in 0..30 {
            let x_params = NormalParams::new(x_pre, x_sigma).unwrap();
            let x = Normal.sample(&x_params, &mut rng).unwrap();
            x_series.append(&mut vec![x]);
            x_pre = x;
            let y_params = NormalParams::new(x, y_sigma).unwrap();
            let y = Normal.sample(&y_params, &mut rng).unwrap();
            y_series.append(&mut vec![y]);
        }
        // estimation by particlefilter
        let distr_x = Normal.condition(&|x: &f64| NormalParams::new(*x, x_sigma));
        let distr_y = Normal.condition(&|y: &f64| NormalParams::new(*y, y_sigma));
        let proposal = MultivariateNormal.condition(&|xy: &(Vec<f64>, Vec<f64>)| {
            ExactMultivariateNormalParams: new(mat!(xy.0, xy.1), x_sigma)
        });
        let test = ParticleFilter::new(y_series, distr_x, distr_y, proposal);
    }
}
