use crate::{Distribution, DistributionError, RandomVariable};
use rand::prelude::*;

pub struct GibbsSampler<T, D>
where
    T: RandomVariable,
    D: Distribution<T = T, U = Vec<T>>,
{
    distributions: Vec<D>,
}

impl<T, D> GibbsSampler<T, D>
where
    T: RandomVariable,
    D: Distribution<T = T, U = Vec<T>>,
{
    pub fn new(distributions: Vec<D>) -> Self {
        Self { distributions }
    }

    pub fn step_sample(
        &self,
        mut data: Vec<T>,
        rng: &mut StdRng,
    ) -> Result<Vec<T>, DistributionError> {
        let n = self.distributions.len();

        let mut shuffled = (0..n).into_iter().collect::<Vec<_>>();
        shuffled.shuffle(rng);

        for i in shuffled {
            let mut condition = data.clone();
            condition.remove(i);

            data[i] = self.distributions[i].sample(&condition, rng)?;
        }

        Ok(data)
    }
}
