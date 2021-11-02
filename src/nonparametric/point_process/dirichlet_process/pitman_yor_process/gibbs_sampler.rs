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
    s: &'a ClusterSwitch,
    value: &'a [T],
    likelihood: &'a SwitchedDistribution<'a, L, T, U>,
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
        s: &'a ClusterSwitch,
        value: &'a [T],
        likelihood: &'a SwitchedDistribution<L, T, U>,
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
    ) -> Result<(usize, u32, U), DistributionError> {
        let n = self.s.s().len();
        let remove_index = rng.gen_range(0..n);

        let likelihood_condition = {
            move |s: &u32| {
                if *s != 0 {
                    Ok(SwitchedParams::Key(*s))
                } else {
                    Ok(SwitchedParams::None)
                }
            }
        };
        let likelihood = self.likelihood.clone().condition(&likelihood_condition);
        let prior_condition = self.gibbs_condition(remove_index);
        let prior = PitmanYorGibbs::new().condition(&prior_condition);

        println!("aa");

        let ds_sampler = DiscreteSliceSampler::new(
            &self.value[remove_index],
            &likelihood,
            &prior,
            self.s
                .s_inv()
                .par_iter()
                .map(|(&si, _)| si)
                .chain(rayon::iter::once(0))
                .collect::<HashSet<u32>>(),
        )?;

        let si = ds_sampler.sample(3, None, rng)?;

        println!("bb");

        let x_in_k = self
            .s
            .s_inv()
            .get(&si)
            .unwrap_or(&HashSet::new())
            .par_iter()
            .filter(|&&i| i != remove_index)
            .map(|&i| self.value[i].clone())
            .chain(rayon::iter::once(self.value[remove_index].clone()))
            .collect::<Vec<_>>();

        let x_likelihood = vec![self.likelihood.distribution().clone(); x_in_k.len()]
            .into_iter()
            .joint();

        println!("cc");

        let mh_sampler =
            MetropolisHastingsSampler::new(&x_in_k, &x_likelihood, &self.base.g0.distr, proposal);

        let theta_k = mh_sampler
            .sample(4, self.base.g0.distr.sample(&(), rng)?, rng)
            .unwrap();

        println!("dd");

        Ok((remove_index, si, theta_k))
    }
}
