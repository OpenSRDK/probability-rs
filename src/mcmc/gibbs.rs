use crate::Distribution;
use rand::prelude::StdRng;
use std::{error::Error, fmt::Debug};

pub struct GibbsSampler<'a, T>
where
    T: Clone + Debug,
{
    distributions: Vec<&'a mut dyn Distribution<'a, T>>,
    conditioning: Box<dyn Fn(usize, &T) -> Result<(), Box<dyn Error>>>,
    iter: usize,
}

impl<'a, T> GibbsSampler<'a, T>
where
    T: Clone + Debug,
{
    pub fn new(
        distributions: Vec<&'a mut dyn Distribution<'a, T>>,
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

    pub fn sample(&mut self, rng: &mut StdRng, initial: Vec<T>) -> Result<Vec<T>, Box<dyn Error>> {
        let mut params = initial;

        for _ in 0..self.iter {
            let n = self.distributions.len();
            for i in 0..n {
                let condition = if i == 0 {
                    params[1..].to_vec()
                } else if i == n - 1 {
                    params[..n - 1].to_vec()
                } else {
                    [&params[..i], &params[i + 1..]].concat()
                };
                (self.conditioning)(i, &condition[i])?;
                params[i] = self.distributions[i].sample(rng)?;
            }
        }

        Ok(params)
    }
}
