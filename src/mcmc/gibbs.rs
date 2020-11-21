use crate::{Distribution, RandomVariable};
use rand::prelude::StdRng;
use std::error::Error;

pub struct GibbsSampler<'a, T, U>
where
    T: RandomVariable,
    U: RandomVariable,
{
    distributions: Vec<&'a mut dyn Distribution<T = T, U = U>>,
    conditioning: Box<dyn Fn(usize, &T) -> Result<(), Box<dyn Error>>>,
    iter: usize,
}

impl<'a, T, U> GibbsSampler<'a, T, U>
where
    T: RandomVariable,
    U: RandomVariable,
{
    pub fn new(
        distributions: Vec<&'a mut dyn Distribution<T = T, U = U>>,
        conditioning: Box<dyn Fn(usize, &T) -> Result<(), Box<dyn Error>>>,
    ) -> Self {
        Self {
            distributions,
            conditioning,
            iter: 32,
        }
    }

    pub fn with_iter(mut self, iter: usize) -> Self {
        self.iter = iter;

        self
    }

    pub fn sample(&mut self, theta: &U, rng: &mut StdRng) -> Result<U, Box<dyn Error>> {
        for _ in 0..self.iter {
            let n = self.distributions.len();
            for i in 0..n {
                let value = self.distributions[i].sample(theta, rng)?;
                (self.conditioning)(i, &value)?;
            }
        }

        Ok(theta.clone())
    }
}
