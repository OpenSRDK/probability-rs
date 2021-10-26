use crate::{Distribution, DistributionError, RandomVariable};
use rand::prelude::*;

pub struct GibbsSampler<'a, T, D>
where
    T: RandomVariable + Default,
    D: Distribution<T = T, U = Vec<T>>,
{
    distributions: Vec<&'a D>,
}

impl<'a, T, D> GibbsSampler<'a, T, D>
where
    T: RandomVariable + Default,
    D: Distribution<T = T, U = Vec<T>>,
{
    pub fn new(distributions: Vec<&'a D>) -> Self {
        Self { distributions }
    }

    pub fn sample(&self, iter: usize, rng: &mut StdRng) -> Result<Vec<T>, DistributionError> {
        let n = self.distributions.len();
        let mut samples = vec![T::default(); n];

        for _ in 0..iter {
            let mut shuffled = (0..n).into_iter().collect::<Vec<_>>();
            shuffled.shuffle(rng);

            for i in shuffled {
                let condition = samples
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| i != j)
                    .map(|(_, v)| v.clone())
                    .collect::<Vec<_>>();

                samples[i] = self.distributions[i].sample(&condition, rng)?;
            }
        }

        Ok(samples)
    }
}
