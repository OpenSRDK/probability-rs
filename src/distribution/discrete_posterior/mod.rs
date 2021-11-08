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

    fn weighted(&self, theta: &T) -> Result<Vec<(f64, &U)>, DistributionError> {
        let weighted = self
            .range
            .par_iter()
            .map(|u| -> Result<_, DistributionError> {
                Ok((self.likelihood.fk(theta, u)? * self.prior.fk(u, &())?, u))
            })
            .collect::<Result<Vec<(f64, &U)>, _>>()?;
        Ok(weighted)
    }

    fn index(&self, weighted: &Vec<(f64, &U)>) -> Result<WeightedIndex<f64>, DistributionError> {
        let index = match WeightedIndex::new(weighted.iter().map(|(w, _)| *w)) {
            Ok(v) => v,
            Err(_) => WeightedIndex::new(vec![1.0; weighted.len()]).unwrap(),
        };
        Ok(index)
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
        let weighted = self.weighted(theta)?;

        let index = self.index(&weighted)?.sample(rng);

        Ok(weighted[index].1.clone())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::distribution::Distribution;
    use crate::*;

    #[test]
    fn it_works() {
        let range = vec![true, false].into_iter().collect::<HashSet<_>>();
        // let mut range = HashSet::new();
        // range.insert(true);
        // range.insert(false);
        let model = DiscretePosterior::new(
            Normal.condition(&|x: &bool| NormalParams::new(if *x { 10.0 } else { 0.0 }, 1.0)),
            Bernoulli.condition(&|_x: &()| BernoulliParams::new(0.5)),
            range,
        );

        // println!("{:?}", model.weighted(&1.0).unwrap());
        let true_result = model.fk(&true, &1.0).unwrap();
        let false_result = model.fk(&false, &1.0).unwrap();
        assert!(true_result < false_result);
    }
}
