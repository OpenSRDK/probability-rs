use crate::{Distribution, *};
use rand::distributions::WeightedIndex;
use rand_distr::Distribution as RandDistribution;
use rayon::prelude::*;
use std::{collections::HashSet, hash::Hash, marker::PhantomData};

#[derive(Clone, Debug)]
pub struct DiscretePosterior<L, P, T, U>
where
    L: Distribution<T = T, U = U>,
    P: Distribution<T = U, U = ()>,
    T: RandomVariable,
    U: RandomVariable + Eq + Hash,
{
    likelihood: L,
    prior: P,
    range: HashSet<U>,
    phantom: PhantomData<T>,
}

impl<L, P, T, U> DiscretePosterior<L, P, T, U>
where
    L: Distribution<T = T, U = U>,
    P: Distribution<T = U, U = ()>,
    T: RandomVariable,
    U: RandomVariable + Eq + Hash,
{
    pub fn new(likelihood: L, prior: P, range: HashSet<U>) -> Self {
        Self {
            likelihood,
            prior,
            range,
            phantom: PhantomData,
        }
    }
}

impl<L, P, T, U> Distribution for DiscretePosterior<L, P, T, U>
where
    L: Distribution<T = T, U = U>,
    P: Distribution<T = U, U = ()>,
    T: RandomVariable,
    U: RandomVariable + Eq + Hash,
{
    type T = U;
    type U = T;

    fn fk(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        Ok(self.likelihood.fk(theta, x)? * self.prior.fk(x, &())?)
    }

    fn sample(
        &self,
        theta: &Self::U,
        rng: &mut dyn rand::RngCore,
    ) -> Result<Self::T, DistributionError> {
        let weighted = self
            .range
            .par_iter()
            .map(|u| -> Result<_, DistributionError> {
                // println!(
                //     "{}, {}",
                //     self.likelihood.fk(theta, u)?,
                //     self.prior.fk(u, &())?
                // );
                Ok((self.likelihood.fk(theta, u)? * self.prior.fk(u, &())?, u))
            })
            .collect::<Result<Vec<(f64, &U)>, _>>()?;

        let index =
            match WeightedIndex::new(weighted.iter().map(|(w, _)| *w).collect::<Vec<_>>()) {
                Ok(v) => v,
                Err(_) => WeightedIndex::new(vec![1.0; weighted.len()]).unwrap(),
            }
            .sample(rng);

        Ok(weighted[index].1.clone())
    }
}
