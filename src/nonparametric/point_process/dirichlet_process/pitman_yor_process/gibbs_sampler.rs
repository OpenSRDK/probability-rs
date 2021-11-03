use crate::*;
use crate::{nonparametric::*, Distribution};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;

/// # Pitman-Yor process
#[derive(Clone, Debug)]
pub struct PitmanYorGibbsSampler<'a, L, T, U, G0>
where
    L: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
    G0: Distribution<T = U, U = ()>,
{
    base: &'a PitmanYorProcessParams<G0, U>,
    s: &'a ClusterSwitch<U>,
    value: &'a [T],
    likelihood: L,
}

impl<'a, L, T, U, G0> PitmanYorGibbsSampler<'a, L, T, U, G0>
where
    L: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
    G0: Distribution<T = U, U = ()>,
{
    pub fn new(
        base: &'a PitmanYorProcessParams<G0, U>,
        s: &'a ClusterSwitch<U>,
        value: &'a [T],
        likelihood: L,
    ) -> Self {
        Self {
            base,
            s,
            value,
            likelihood,
        }
    }

    fn gibbs_condition(
        &'a self,
        remove_index: usize,
    ) -> impl Fn(&()) -> Result<PitmanYorGibbsParams<'a, G0, U>, DistributionError> {
        move |_| PitmanYorGibbsParams::new(self.base, self.s, remove_index)
    }

    /// 0 means new cluster. However, you can't use 0 for `s` so use another value which will not conflict.
    pub fn step_sample(
        &self,
        proposal: &impl Distribution<T = U, U = U>,
        rng: &mut dyn RngCore,
    ) -> Result<(usize, PitmanYorGibbsSample, U), DistributionError> {
        let n = self.s.s().len();
        let remove_index = rng.gen_range(0..n);

        let new_theta = self.base.g0.distr.sample(&(), rng)?;

        let likelihood_condition = |s: &PitmanYorGibbsSample| {
            Ok(match *s {
                PitmanYorGibbsSample::Existing(k) => SwitchedParams::Key(k),
                PitmanYorGibbsSample::New => SwitchedParams::Direct(new_theta.clone()),
            })
        };
        let likelihood = self
            .likelihood
            .switch(self.s.theta())
            .condition(&likelihood_condition);
        let prior_condition = self.gibbs_condition(remove_index);
        let prior = PitmanYorGibbs::new().condition(&prior_condition);

        let ds_sampler = DiscreteSliceSampler::new(
            &self.value[remove_index],
            &likelihood,
            &prior,
            &self
                .s
                .s_inv()
                .par_iter()
                .map(|(&si, _)| PitmanYorGibbsSample::Existing(si))
                .chain(rayon::iter::once(PitmanYorGibbsSample::New))
                .collect::<HashSet<PitmanYorGibbsSample>>(),
        )?;

        let si = ds_sampler.sample(3, None, rng)?;

        let theta_k = match si {
            PitmanYorGibbsSample::Existing(k) => {
                let x_in_k = self
                    .s
                    .s_inv()
                    .get(&k)
                    .unwrap_or(&HashSet::new())
                    .par_iter()
                    .filter(|&&i| i != remove_index)
                    .map(|&i| self.value[i].clone())
                    .chain(rayon::iter::once(self.value[remove_index].clone()))
                    .collect::<Vec<_>>();

                let x_likelihood = vec![self.likelihood.clone(); x_in_k.len()]
                    .into_iter()
                    .joint();

                let mh_sampler = MetropolisHastingsSampler::new(
                    &x_in_k,
                    &x_likelihood,
                    &self.base.g0.distr,
                    proposal,
                );

                mh_sampler.sample(4, self.base.g0.distr.sample(&(), rng)?, rng)?
            }
            PitmanYorGibbsSample::New => new_theta,
        };

        Ok((remove_index, si, theta_k))
    }
}
