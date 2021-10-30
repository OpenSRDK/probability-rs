use crate::{DiscreteUniform, Distribution, DistributionError, RandomVariable};
use rand::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

/// Sample b from posterior p(b|a) with likelihood p(a|b) and prior p(b)
pub struct DiscreteSliceSampler<B>
where
    B: RandomVariable + Eq + Hash,
{
    range: HashMap<B, f64>,
}

#[derive(thiserror::Error, Debug)]
pub enum SliceSamplingError {
    #[error("out of range")]
    OutOfRange,
    #[error("Unknown error")]
    Unknown,
}

impl<B> DiscreteSliceSampler<B>
where
    B: RandomVariable + Eq + Hash,
{
    pub fn new<L, P, A>(
        value: &A,
        likelihood: &L,
        prior: &P,
        range: &HashSet<B>,
    ) -> Result<Self, DistributionError>
    where
        L: Distribution<T = A, U = B>,
        P: Distribution<T = B, U = ()>,
        A: RandomVariable,
    {
        let range = range
            .iter()
            .map(|b| -> Result<_, DistributionError> {
                Ok((b.clone(), likelihood.p(value, b)? * prior.p(b, &())?))
            })
            .collect::<Result<HashMap<_, _>, _>>()?;

        Ok(Self { range })
    }

    pub fn sample(
        &self,
        iter: usize,
        initial: Option<&B>,
        rng: &mut StdRng,
    ) -> Result<B, DistributionError> {
        let mut u = match initial {
            Some(initial) => {
                let initial_f = match self.range.get(initial) {
                    Some(&v) => Ok(v),
                    None => Err(DistributionError::InvalidParameters(
                        SliceSamplingError::OutOfRange.into(),
                    )),
                }?;
                rng.gen_range(0.0..initial_f)
            }
            None => 0.0,
        };
        let mut count = 0;
        let mut last: B;

        loop {
            let range = self
                .range
                .iter()
                .filter(|&(_, &v)| v > u)
                .map(|(k, _)| k.clone())
                .collect::<HashSet<B>>();

            last = DiscreteUniform::new().sample(&range, rng)?;
            let last_f = *self.range.get(&last).unwrap();
            u = rng.gen_range(0.0..last_f);

            count += 1;
            if iter <= count {
                break;
            }
        }

        Ok(last)
    }
}
