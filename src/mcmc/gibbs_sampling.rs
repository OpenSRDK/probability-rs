use crate::{Distribution, DistributionError, RandomVariable};
use rand::prelude::*;

pub struct GibbsSampler;

impl GibbsSampler {
    /// https://qiita.com/masasora/items/5469638d93d9c834724b
    pub fn step_sample<D, T>(
        &self,
        index: usize,
        data: &[T],
        distr: D,
        rng: &mut StdRng,
    ) -> Result<T, DistributionError>
    where
        D: Distribution<T = T, U = Vec<T>>,
        T: RandomVariable,
    {
        let mut condition = data.to_vec();
        condition.remove(index);

        let data = distr.sample(&condition, rng)?;

        Ok(data)
    }
}
