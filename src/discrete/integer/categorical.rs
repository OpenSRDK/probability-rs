use crate::*;
use crate::{Distribution, DistributionError};
use rand::prelude::*;
use rand_distr::num_traits::Pow;
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
    type T = usize;
    type U = CategoricalParams;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let k = *x;
        if k < theta.p().len() {
            return Err(DistributionError::InvalidParameters(
                CategoricalError::IndexOutOfRange.into(),
            ));
        }
        Ok(theta.p()[k])
    }

    fn sample(
        &self,
        theta: &Self::U,
        rng: &mut dyn rand::RngCore,
    ) -> Result<Self::T, DistributionError> {
        let u = rng.gen_range(0.0..=1.0);
        let mut p_sum = 0.0;
        for i in 0..theta.p().len() {
            p_sum += theta.p()[i];
            if u <= p_sum {
                return Ok(i);
            }
        }
        Err(DistributionError::Others(CategoricalError::Unknown.into()))
    }
}

#[derive(Clone, Debug)]
pub struct CategoricalParams {
    p: Vec<f64>,
}

impl CategoricalParams {
    pub fn new(p: Vec<f64>) -> Result<Self, DistributionError> {
        let mut p_sum = 0.0;
        for &pi in p.iter() {
            if pi < 0.0 || 1.0 < pi {
                return Err(DistributionError::InvalidParameters(
                    CategoricalError::PMustBeProbability.into(),
                ));
            }
            p_sum += pi;
        }
        let epsilon = 10.0.pow(-6);
        if p_sum < 1.0 - epsilon || 1.0 + epsilon < p_sum {
            return Err(DistributionError::InvalidParameters(
                CategoricalError::SumOfPMustBeOne.into(),
            ));
        }

        Ok(Self { p })
    }

    pub fn p(&self) -> &Vec<f64> {
        &self.p
    }
}

impl<Rhs, TRhs> Mul<Rhs> for Categorical
where
    Rhs: Distribution<T = TRhs, U = CategoricalParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, usize, TRhs, CategoricalParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for Categorical
where
    Rhs: Distribution<T = CategoricalParams, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, usize, CategoricalParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Categorical, CategoricalParams, Distribution};
    use rand::prelude::*;
    #[test]
    fn it_works() {
        let mut rng = StdRng::from_seed([1; 32]);

        let p = vec![0.1, 0.2, 0.3, 0.4];
        let theta = CategoricalParams::new(p).unwrap();

        let hoge = Categorical.sample(&theta, &mut rng).unwrap();
        assert_eq!(hoge, 2);
    }
}
