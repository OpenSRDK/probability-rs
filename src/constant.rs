use crate::{Distribution, IndependentJoint, RandomVariable};
use rand::prelude::StdRng;
use std::{error::Error, ops::Mul};

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

impl<R, T, TR> Mul<R> for Constant<T>
where
    R: Distribution<T = TR, U = ()>,
    T: RandomVariable,
    TR: RandomVariable,
{
    type Output = IndependentJoint<Self, R, T, TR, ()>;

    fn mul(self, rhs: R) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}
