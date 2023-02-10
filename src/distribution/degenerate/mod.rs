use crate::{Distribution, RandomVariable, SamplableDistribution};

#[derive(Clone, Debug)]
pub struct Degenerate<T>
where
    T: RandomVariable + PartialEq,
{
    value: T,
}

impl<T> Degenerate<T>
where
    T: RandomVariable + PartialEq,
{
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T> Distribution for Degenerate<T>
where
    T: RandomVariable + PartialEq,
{
    type Value = T;
    type Condition = ();

    fn p_kernel(
        &self,
        x: &Self::Value,
        _theta: &Self::Condition,
    ) -> Result<f64, crate::DistributionError> {
        if self.value.eq(x) {
            Ok(1.0)
        } else {
            Ok(0.0)
        }
    }
}

impl<T> SamplableDistribution for Degenerate<T>
where
    T: RandomVariable + PartialEq,
{
    fn sample(
        &self,
        _theta: &Self::Condition,
        _rng: &mut dyn rand::RngCore,
    ) -> Result<Self::Value, crate::DistributionError> {
        Ok(self.value.clone())
    }
}
