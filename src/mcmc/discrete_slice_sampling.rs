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
    f_map: HashMap<B, f64>,
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
        let f_map = range
            .iter()
            .map(|b| -> Result<_, DistributionError> {
                Ok((b.clone(), likelihood.fk(value, b)? * prior.fk(b, &())?))
            })
            .collect::<Result<HashMap<_, _>, _>>()?;

        Ok(Self { f_map })
    }

    pub fn sample(
        &self,
        iter: usize,
        initial: Option<&B>,
        rng: &mut dyn RngCore,
    ) -> Result<B, DistributionError> {
        let initial_f = match initial {
            Some(initial) => match self.f_map.get(initial) {
                Some(&v) => Ok(v),
                None => Err(DistributionError::InvalidParameters(
                    SliceSamplingError::OutOfRange.into(),
                )),
            }?,
            None => 0.0,
        };
        let mut count = 0;
        let mut last: B;
        let mut last_f = initial_f;

        loop {
            let u = rng.gen_range(0.0..=last_f);
            let range = self
                .f_map
                .iter()
                .filter(|&(_, &v)| u <= v)
                .map(|(k, _)| k.clone())
                .collect::<HashSet<B>>();

            last = DiscreteUniform::new().sample(&range, rng)?;
            last_f = *self.f_map.get(&last).unwrap();

            count += 1;
            if iter <= count {
                break;
            }
        }

        Ok(last)
    }
}
