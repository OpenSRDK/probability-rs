pub mod params;

pub use params::*;

use crate::*;
use crate::{Distribution, DistributionError};
use rand::distributions::WeightedIndex;
use rand_distr::Distribution as RandDistribution;
use std::ops::{BitAnd, Mul};

#[derive(Clone, Debug)]
pub struct Categorical;

#[derive(thiserror::Error, Debug)]
pub enum CategoricalError {
    #[error("'p' must be probability.")]
    PMustBeProbability,
    #[error("Sum of 'p' must be 1.")]
    SumOfPMustBeOne,
    #[error("Index is out of range.")]
    IndexOutOfRange,
    #[error("Unknown.")]
    Unknown,
}

impl Distribution for Categorical {
    type Value = usize;
    type Condition = CategoricalParams;

    fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let k = *x;
        if k < theta.p().len() {
            return Err(DistributionError::InvalidParameters(
                CategoricalError::IndexOutOfRange.into(),
            ));
        }
        Ok(theta.p()[k])
    }
}

impl DiscreteDistribution for Categorical {}

impl<Rhs, TRhs> Mul<Rhs> for Categorical
where
    Rhs: Distribution<Value = TRhs, Condition = CategoricalParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, usize, TRhs, CategoricalParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for Categorical
where
    Rhs: Distribution<Value = CategoricalParams, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, usize, CategoricalParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

impl SampleableDistribution for Categorical {
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn rand::RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let index = match WeightedIndex::new(theta.p().clone()) {
            Ok(v) => Ok(v),
            Err(e) => Err(DistributionError::Others(e.into())),
        }?
        .sample(rng);

        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Categorical, CategoricalParams, Distribution, SampleableDistribution};
    use rand::prelude::*;
    #[test]
    fn it_works() {
        let mut rng = StdRng::from_seed([1; 32]);

        // {
        //     // 0.17522173557509294
        //     println!("{}", rng.gen_range(0.0..=1.0));
        //     return;
        // }
        let p = vec![0.1, 0.2, 0.3, 0.4];
        let theta = CategoricalParams::new(p).unwrap();

        let hoge = Categorical.sample(&theta, &mut rng).unwrap();
        assert_eq!(hoge, 1);
    }
}
