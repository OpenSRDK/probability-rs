use crate::rand::SeedableRng;
use crate::{
    ContinuousSamplesDistribution, Distribution, DistributionError, RandomVariable,
    SamplableDistribution,
};
use rand::rngs::StdRng;

// Sampling Importance Resampling
pub struct ParticleFilter<Y, X, DY, DX, PD>
where
    Y: RandomVariable,
    X: RandomVariable + PartialEq,
    DY: Distribution<Value = Y, Condition = X>,
    DX: Distribution<Value = X, Condition = X>,
    PD: SamplableDistribution<Value = X, Condition = (Vec<X>, Vec<Y>)>,
{
    observable: Vec<Y>,
    distr_x: DX,
    distr_y: DY,
    proposal: PD,
}

impl<Y, X, DY, DX, PD> ParticleFilter<Y, X, DY, DX, PD>
where
    Y: RandomVariable,
    X: RandomVariable + PartialEq,
    DY: Distribution<Value = Y, Condition = X>,
    DX: Distribution<Value = X, Condition = X>,
    PD: SamplableDistribution<Value = X, Condition = (Vec<X>, Vec<Y>)>,
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
        let mut rng = StdRng::from_seed([13; 32]);

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

        for t in 0..self.observable.len() {
            let mut p = (0..particles_len)
                .into_iter()
                .map(|i| -> Result<_, DistributionError> {
                    let pi = self.proposal.sample(
                        &(
                            (&vecvec_p[i][0..t + 1]).to_vec(),
                            (self.observable[0..t + 1]).to_vec(),
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
                            * self.distr_y.p_kernel(&self.observable[t], &p[i])?
                            * self.distr_x.p_kernel(&p[i], &p_previous[i])?
                            / self.proposal.p_kernel(
                                &p[i],
                                &(
                                    (vecvec_p[i][0..t + 1]).to_vec(),
                                    (self.observable[0..t + 1]).to_vec(),
                                ),
                            )?;
                        Ok(wi_orig)
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let w = (0..particles_len)
                    .into_iter()
                    .map(|i| -> Result<_, DistributionError> {
                        let wi = w_orig[i] / (w_orig.iter().map(|wi_orig| wi_orig).sum::<f64>());
                        Ok(wi)
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let eff = 1.0 / (w.iter().map(|wi| wi.powi(2)).sum::<f64>());
                println!("{:#?}", eff);

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

                    vecvec_p = (0..particles_len)
                        .into_iter()
                        .map(|i| -> Result<_, DistributionError> {
                            let mut vec_pi = vec![p[i].clone()];
                            vecvec_p[i].append(&mut vec_pi);
                            let vecvec_pi = vecvec_p[i].clone();
                            Ok(vecvec_pi)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
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

#[cfg(test)]
mod tests {
    use crate::distribution::Distribution;
    use crate::*;
    use rand::prelude::*;
    use sir::ParticleFilter;

    #[test]
    fn it_works() {
        // create test data
        let x_sigma = 2.0;
        let y_sigma = 4.0;
        let time = 30;
        let mut rng = StdRng::from_seed([5; 32]);
        let mut x_series = Vec::new();
        let mut x_pre = 0.0;
        let mut y_series = Vec::new();
        for _i in 0..time {
            let x_params = NormalParams::new(x_pre, x_sigma).unwrap();
            let x = Normal.sample(&x_params, &mut rng).unwrap();
            x_series.append(&mut vec![x]);
            x_pre = x;
            let y_params = NormalParams::new(x.powi(3) * 0.3, y_sigma).unwrap();
            let y = Normal.sample(&y_params, &mut rng).unwrap();
            y_series.append(&mut vec![y]);
        }
        // estimation by particlefilter
        let p_num = 100;
        let fn_x = |x: &f64| NormalParams::new(*x, x_sigma);
        let fn_y = |y: &f64| NormalParams::new(y.powi(3) * 0.3, y_sigma);
        let fn_p = |xy: &(Vec<f64>, Vec<f64>)| NormalParams::new(xy.0[xy.0.len() - 1], x_sigma);
        let distr_x = Normal.map_condition(&fn_x);
        let distr_y = Normal.map_condition(&fn_y);
        let proposal = Normal.map_condition(&fn_p);
        let test = ParticleFilter::new(y_series.clone(), distr_x, distr_y, proposal).unwrap();
        let p_initial = (0..p_num)
            .into_iter()
            .map(|_i| -> Result<_, DistributionError> {
                let p = Normal
                    .sample(&NormalParams::new(x_pre, x_sigma).unwrap(), &mut rng)
                    .unwrap();
                Ok(p)
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let thr = p_num as f64 * 0.99;
        let result = &test.filtering(p_initial, thr).unwrap();
        let est_x = (0..time)
            .into_iter()
            .map(|i| -> Result<_, DistributionError> {
                let a: f64 = result[i].samples().iter().sum();
                let b: f64 = result[i].samples().len() as f64;
                let x = a / b;
                Ok(x)
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        println!("{:#?}", x_series);
        println!("{:#?}", x_series.len());
        println!("{:#?}", y_series);
        println!("{:#?}", y_series.len());
        println!("{:#?}", est_x);
        println!("{:#?}", est_x.len());
    }
}
