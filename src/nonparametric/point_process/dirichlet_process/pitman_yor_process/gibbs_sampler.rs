use crate::*;
use crate::{nonparametric::*, Distribution};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;

/// # Pitman-Yor process
#[derive(Clone, Debug)]
pub struct PitmanYorGibbsSampler<'a, T, L>
where
    T: RandomVariable,
    L: Distribution<T = T, U = u32>,
{
    base: &'a PitmanYorProcessParams,
    s: &'a ClusterSwitch,
    value: &'a [T],
    likelihood: &'a L,
}

impl<'a, T, L> PitmanYorGibbsSampler<'a, T, L>
where
    T: RandomVariable,
    L: Distribution<T = T, U = u32>,
{
    pub fn new(
        base: &'a PitmanYorProcessParams,
        s: &'a ClusterSwitch,
        value: &'a [T],
        likelihood: &'a L,
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
    ) -> impl Fn(&()) -> Result<PitmanYorGibbsParams<'a>, DistributionError> {
        move |_| PitmanYorGibbsParams::new(self.base, self.s, remove_index)
    }

    /// 0 means new cluster. However, you can't use 0 for `s` so use another value which will not conflict.
    pub fn sample(
        &self,
        seed: Option<&(dyn Fn(usize) -> [u8; 32] + Send + Sync)>,
    ) -> Result<ClusterSwitch, DistributionError> {
        let n = self.s.s().len();

        let s = (0..n)
            .into_par_iter()
            .map(|i| -> Result<u32, DistributionError> {
                let condition = self.gibbs_condition(i);
                let prior = PitmanYorGibbs::new().condition(&condition);

                let ds_sampler = DiscreteSliceSampler::new(
                    &self.value[i],
                    self.likelihood,
                    &prior,
                    self.s
                        .s_inv()
                        .par_iter()
                        .map(|(&si, _)| si)
                        .chain(rayon::iter::once(0))
                        .collect::<HashSet<u32>>(),
                )?;

                let mut rng: Box<dyn RngCore> = match seed {
                    Some(f) => Box::new(StdRng::from_seed(f(i))),
                    None => Box::new(thread_rng()),
                };

                let si = ds_sampler.sample(3, None, &mut rng)?;

                Ok(si)
            })
            .collect::<Result<Vec<_>, _>>()?;

        ClusterSwitch::new(s)
    }
}
