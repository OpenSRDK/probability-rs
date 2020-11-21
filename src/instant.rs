use crate::{Distribution, RandomVariable};
use rand::prelude::StdRng;
use std::error::Error;

pub struct InstantDistribution<'a, T, U>
where
    T: RandomVariable,
    U: RandomVariable,
{
    p: &'a dyn Fn(&T, &U) -> Result<f64, Box<dyn Error>>,
    sample: &'a dyn Fn(&U, &mut StdRng) -> Result<T, Box<dyn Error>>,
}

impl<'a, T, U> InstantDistribution<'a, T, U>
where
    T: RandomVariable,
    U: RandomVariable,
{
    pub fn new(
        p: &'a dyn Fn(&T, &U) -> Result<f64, Box<dyn Error>>,
        sample: &'a dyn Fn(&U, &mut StdRng) -> Result<T, Box<dyn Error>>,
    ) -> Self {
        Self { p, sample }
    }
}

impl<'a, T, U> Distribution for InstantDistribution<'a, T, U>
where
    T: RandomVariable,
    U: RandomVariable,
{
    type T = T;
    type U = U;

    fn p(&self, x: &T, theta: &U) -> Result<f64, Box<dyn Error>> {
        (self.p)(x, theta)
    }

    fn sample(&self, theta: &U, rng: &mut StdRng) -> Result<T, Box<dyn Error>> {
        (self.sample)(theta, rng)
    }
}
