use crate::Distribution;
use rand::prelude::StdRng;
use std::error::Error;

pub struct InstantDistribution<T> {
    p: Box<dyn Fn(&T) -> Result<f64, Box<dyn Error>>>,
    sample: Box<dyn Fn(&mut StdRng) -> Result<T, Box<dyn Error>>>,
}

impl<T> InstantDistribution<T> {
    pub fn new(
        p: Box<dyn Fn(&T) -> Result<f64, Box<dyn Error>>>,
        sample: Box<dyn Fn(&mut StdRng) -> Result<T, Box<dyn Error>>>,
    ) -> Self {
        Self { p, sample }
    }
}

impl<T> Distribution<T> for InstantDistribution<T> {
    fn p(&self, x: &T) -> Result<f64, Box<dyn Error>> {
        (self.p)(x)
    }

    fn sample(&self, rng: &mut StdRng) -> Result<T, Box<dyn Error>> {
        (self.sample)(rng)
    }
}
