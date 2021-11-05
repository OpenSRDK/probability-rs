use std::collections::{HashMap, HashSet};

use crate::*;
use crate::{nonparametric::*, Distribution};
use rand::prelude::*;
use rayon::prelude::*;

/// # Pitman-Yor process
pub struct PitmanYorGibbsSampler<'a, L, T, U, G0>
where
    L: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
    G0: Distribution<T = U, U = ()>,
{
    base: &'a PitmanYorProcessParams<G0, U>,
    switch: &'a ClusterSwitch<U>,
    x: &'a [T],
    likelihood: &'a L,
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
        likelihood: &'a L,
    ) -> Self {
        Self {
            base,
            switch: s,
            x: value,
            likelihood,
        }
    }

    fn gibbs_condition(
        &'a self,
        s_inv: &'a HashMap<u32, HashSet<usize>>,
        n: usize,
    ) -> impl Fn(&()) -> Result<PitmanYorGibbsParams<'a, G0, U>, DistributionError> {
        move |_| Ok(PitmanYorGibbsParams::new(self.base, s_inv, n))
    }

    fn sample_s(
        &self,
        x: &T,
        new_theta: &U,
        switch: &ClusterSwitch<U>,
        rng: &mut dyn RngCore,
    ) -> Result<PitmanYorGibbsSample, DistributionError> {
        let likelihood_condition = |s: &PitmanYorGibbsSample| {
            Ok(match *s {
                PitmanYorGibbsSample::Existing(k) => SwitchedParams::Key(k),
                PitmanYorGibbsSample::New => SwitchedParams::Direct(new_theta.clone()),
            })
        };

        let likelihood = self
            .likelihood
            .switch(switch.theta())
            .condition(&likelihood_condition);
        let prior_condition = self.gibbs_condition(switch.s_inv(), switch.s().len());
        let prior = PitmanYorGibbs::new().condition(&prior_condition);

        let posterior = DiscretePosterior::new(
            likelihood,
            prior,
            switch
                .s_inv()
                .par_iter()
                .map(|(&si, _)| PitmanYorGibbsSample::Existing(si))
                .chain(rayon::iter::once(PitmanYorGibbsSample::New))
                .collect::<HashSet<PitmanYorGibbsSample>>(),
        );

        posterior.sample(x, rng)
    }

    fn sample_theta(
        &self,
        x_in_k: &Vec<T>,
        proposal: &impl Distribution<T = U, U = U>,
    ) -> Result<U, DistributionError> {
        let x_likelihood = vec![self.likelihood.clone(); x_in_k.len()]
            .into_iter()
            .joint();

        let mh_sampler =
            MetropolisHastingsSampler::new(x_in_k, &x_likelihood, &self.base.g0.distr, proposal);
        let mut rng = thread_rng();

        mh_sampler.sample(4, self.base.g0.distr.sample(&(), &mut rng)?, &mut rng)
    }

    pub fn step_sample(
        &self,
        proposal: &impl Distribution<T = U, U = U>,
        rng: &mut dyn RngCore,
    ) -> Result<ClusterSwitch<U>, DistributionError> {
        let n = self.switch.s().len();

        let mut ret = self.switch.clone();

        for remove_index in 0..n {
            let new_theta = self.base.g0.distr.sample(&(), rng)?;

            ret.remove(remove_index);

            let sampled = self.sample_s(&self.x[remove_index], &new_theta, &ret, rng)?;

            let si = ret.set_s(remove_index, sampled);
            if !ret.theta().contains_key(&si) {
                ret.theta_mut().insert(si, new_theta);
            }
        }

        *ret.theta_mut() = ret
            .s_inv()
            .par_iter()
            .map(|(&k, indice)| -> Result<_, DistributionError> {
                let x_in_k = indice
                    .par_iter()
                    .map(|&i| self.x[i].clone())
                    .collect::<Vec<_>>();

                let theta_k = self.sample_theta(&x_in_k, proposal)?;

                Ok((k, theta_k))
            })
            .collect::<Result<_, _>>()?;

        Ok(ret)
    }
}