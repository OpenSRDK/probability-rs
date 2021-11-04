use std::collections::HashSet;

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
    switch: &'a mut ClusterSwitch<U>,
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
        s: &'a mut ClusterSwitch<U>,
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
        remove_index: usize,
    ) -> impl Fn(&()) -> Result<PitmanYorGibbsParams<'a, G0, U>, DistributionError> {
        move |_| PitmanYorGibbsParams::new(self.base, self.switch, remove_index)
    }

    pub fn sample(
        &mut self,
        proposal: &impl Distribution<T = U, U = U>,
        rng: &mut dyn RngCore,
    ) -> Result<(), DistributionError> {
        let n = self.switch.s().len();

        for remove_index in 0..n {
            let new_theta = self.base.g0.distr.sample(&(), rng)?;
            let sampled = {
                let likelihood_condition = |s: &PitmanYorGibbsSample| {
                    Ok(match *s {
                        PitmanYorGibbsSample::Existing(k) => SwitchedParams::Key(k),
                        PitmanYorGibbsSample::New => SwitchedParams::Direct(new_theta.clone()),
                    })
                };

                let likelihood = self
                    .likelihood
                    .switch(self.switch.theta())
                    .condition(&likelihood_condition);
                let prior_condition = self.gibbs_condition(remove_index);
                let prior = PitmanYorGibbs::new().condition(&prior_condition);

                let posterior = DiscretePosterior::new(
                    likelihood,
                    prior,
                    self.switch
                        .theta()
                        .par_iter()
                        .map(|(&si, _)| PitmanYorGibbsSample::Existing(si))
                        .chain(rayon::iter::once(PitmanYorGibbsSample::New))
                        .collect::<HashSet<PitmanYorGibbsSample>>(),
                );

                posterior.sample(&self.x[remove_index], rng)?
            };

            let si = self.switch.set_s(remove_index, sampled);
            if let PitmanYorGibbsSample::New = sampled {
                self.switch.theta_mut().insert(si, new_theta);
            }
        }

        *self.switch.theta_mut() = self
            .switch
            .s_inv()
            .par_iter()
            .map(|(&k, indice)| -> Result<_, DistributionError> {
                let x_in_k = indice
                    .par_iter()
                    .map(|&i| self.x[i].clone())
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
                let mut rng = StdRng::from_seed([1; 32]);

                let theta_k =
                    mh_sampler.sample(4, self.base.g0.distr.sample(&(), &mut rng)?, &mut rng)?;

                Ok((k, theta_k))
            })
            .collect::<Result<_, _>>()?;

        Ok(())
    }
}
