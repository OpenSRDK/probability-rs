use crate::{Distribution, DistributionParamVal, LogDistribution};
use rand::prelude::StdRng;
use std::{error::Error, fmt::Debug};

pub struct InstantDistribution<'a, T>
where
    T: Clone + Debug,
{
    p: Box<dyn Fn(&T) -> Result<f64, Box<dyn Error>>>,
    sample: Box<dyn Fn(&mut StdRng) -> Result<T, Box<dyn Error>>>,
    ln: Box<dyn Fn(&'a mut DistributionParamVal<T>) -> LogDistribution<'a>>,
}

impl<'a, T> InstantDistribution<'a, T>
where
    T: Clone + Debug,
{
    pub fn new(
        p: Box<dyn Fn(&T) -> Result<f64, Box<dyn Error>>>,
        sample: Box<dyn Fn(&mut StdRng) -> Result<T, Box<dyn Error>>>,
        ln: Box<dyn Fn(&'a mut DistributionParamVal<T>) -> LogDistribution<'a>>,
    ) -> Self {
        Self { p, sample, ln }
    }
}

impl<'a, T> Distribution<'a, T> for InstantDistribution<'a, T>
where
    T: Clone + Debug,
{
    fn p(&self, x: &T) -> Result<f64, Box<dyn Error>> {
        (self.p)(x)
    }

    fn sample(&self, rng: &mut StdRng) -> Result<T, Box<dyn Error>> {
        (self.sample)(rng)
    }

    fn ln(&'a mut self, x: &'a mut DistributionParamVal<T>) -> LogDistribution<'a> {
        (self.ln)(x)
    }
}
