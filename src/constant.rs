use crate::{Distribution, IndependentJoint, RandomVariable};
use rand::prelude::StdRng;
use std::{error::Error, ops::Mul};

/// # Constant
/// ![tex](https://latex.codecogs.com/svg.latex?p%28x%29%3D\delta%28x-x%5E*%29)
pub struct Constant<T>
where
    T: RandomVariable,
{
    value: T,
}

impl<T> Constant<T>
where
    T: RandomVariable,
{
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T> Distribution for Constant<T>
where
    T: RandomVariable,
{
    type T = T;
    type U = ();

    fn p(&self, x: &T, _: &()) -> Result<f64, Box<dyn Error>> {
        if self.value.eq(x) {
            Ok(1.0)
        } else {
            Ok(0.0)
        }
    }

    fn sample(&self, _: &(), _: &mut StdRng) -> Result<T, Box<dyn Error>> {
        Ok(self.value.clone())
    }
}

impl<T, Rhs, TRhs> Mul<Rhs> for Constant<T>
where
    Rhs: Distribution<T = TRhs, U = ()>,
    T: RandomVariable,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, T, TRhs, ()>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}
