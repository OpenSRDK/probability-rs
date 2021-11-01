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
    pub fn sample(
        &self,
        seed: Option<&(dyn Fn(usize) -> [u8; 32] + Send + Sync)>,
    ) -> Result<ClusterSwitch, DistributionError> {
        let n = self.s.s().len();

        let mut clusters = self.s.clone();

        let s = (0..n)
            .into_par_iter()
            .map(|i| -> Result<u32, DistributionError> {
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
                let prior_condition = self.gibbs_condition(i);
                let prior = PitmanYorGibbs::new().condition(&prior_condition);

                let mut rng: Box<dyn RngCore> = match seed {
                    Some(f) => Box::new(StdRng::from_seed(f(i))),
                    None => Box::new(thread_rng()),
                };

                let ds_sampler = DiscreteSliceSampler::new(
                    &self.value[i],
                    &likelihood,
                    &prior,
                    self.s
                        .s_inv()
                        .par_iter()
                        .map(|(&si, _)| si)
                        .chain(rayon::iter::once(0))
                        .collect::<HashSet<u32>>(),
                )?;

                let si = ds_sampler.sample(3, None, &mut rng)?;

                Ok(si)
            })
            .collect::<Result<Vec<_>, _>>()?;

        for (i, si) in s.into_iter().enumerate() {
            clusters.set_s(i, si);
        }

        Ok(clusters)
    }
}
