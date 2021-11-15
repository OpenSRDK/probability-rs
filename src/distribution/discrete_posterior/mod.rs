use crate::{Distribution, *};
use rand::distributions::WeightedIndex;
use rand_distr::Distribution as RandDistribution;
use rayon::prelude::*;
use std::{collections::HashSet, hash::Hash, marker::PhantomData};

#[derive(Clone, Debug)]
pub struct DiscretePosterior<L, P, A, B>
where
    L: Distribution<Value = A, Condition = B>,
    P: Distribution<Value = B, Condition = ()>,
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
    L: Distribution<Value = A, Condition = B>,
    P: Distribution<Value = B, Condition = ()>,
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

    fn weighted(&self, theta: &A) -> Result<Vec<(f64, &B)>, DistributionError> {
        let weighted = self
            .range
            .par_iter()
            .map(|u| -> Result<_, DistributionError> {
                Ok((self.likelihood.fk(theta, u)? * self.prior.fk(u, &())?, u))
            })
            .collect::<Result<Vec<(f64, &B)>, _>>()?;
        Ok(weighted)
    }

    fn index(&self, weighted: &Vec<(f64, &B)>) -> Result<WeightedIndex<f64>, DistributionError> {
        let index = match WeightedIndex::new(weighted.iter().map(|(w, _)| *w)) {
            Ok(v) => v,
            Err(_) => WeightedIndex::new(vec![1.0; weighted.len()]).unwrap(),
        };
        Ok(index)
    }
}

impl<L, P, A, B> Distribution for DiscretePosterior<L, P, A, B>
where
    L: Distribution<Value = A, Condition = B>,
    P: Distribution<Value = B, Condition = ()>,
    A: RandomVariable,
    B: RandomVariable + Eq + Hash,
{
    type Value = B;
    type Condition = A;

    fn fk(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        Ok(self.likelihood.fk(theta, x)? * self.prior.fk(x, &())?)
    }

    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn rand::RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let weighted = self
            .range
            .par_iter()
            .map(|u| -> Result<_, DistributionError> {
                Ok((self.likelihood.fk(theta, u)? * self.prior.fk(u, &())?, u))
            })
            .collect::<Result<Vec<(f64, &B)>, _>>()?;

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
