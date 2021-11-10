use crate::{Distribution, *};
use rand::distributions::WeightedIndex;
use rand_distr::Distribution as RandDistribution;
use rayon::prelude::*;
use std::{collections::HashSet, hash::Hash, marker::PhantomData};

#[derive(Clone, Debug)]
pub struct DiscretePosterior<L, P, A, B>
where
    L: Distribution<T = A, U = B>,
    P: Distribution<T = B, U = ()>,
    A: RandomVariable,
    B: RandomVariable + Eq + Hash,
{
    likelihood: L,
    prior: P,
    range: HashSet<B>,
    phantom: PhantomData<A>,
}

impl<L, P, A, B> DiscretePosterior<L, P, A, B>
where
    L: Distribution<T = A, U = B>,
    P: Distribution<T = B, U = ()>,
    A: RandomVariable,
    B: RandomVariable + Eq + Hash,
{
    pub fn new(likelihood: L, prior: P, range: HashSet<B>) -> Self {
        Self {
            likelihood,
            prior,
            range,
            phantom: PhantomData,
        }
    }
}

impl<L, P, A, B> Distribution for DiscretePosterior<L, P, A, B>
where
    L: Distribution<T = A, U = B>,
    P: Distribution<T = B, U = ()>,
    A: RandomVariable,
    B: RandomVariable + Eq + Hash,
{
    type T = B;
    type U = A;

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
                Ok((self.likelihood.fk(theta, u)? * self.prior.fk(u, &())?, u))
            })
            .collect::<Result<Vec<(f64, &B)>, _>>()?;

        let index = match WeightedIndex::new(weighted.iter().map(|(w, _)| *w)) {
            Ok(v) => v,
            Err(_) => WeightedIndex::new(vec![1.0; weighted.len()]).unwrap(),
        }
        .sample(rng);

        Ok(weighted[index].1.clone())
    }
}
