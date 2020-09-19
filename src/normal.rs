use crate::Distribution;
use rand::prelude::*;
use rand_distr::Normal as RandNormal;
use std::{error::Error, f64::consts::PI};

#[derive(Debug)]
pub struct Normal {
    mean: f64,
    std_dev: f64,
}

#[derive(thiserror::Error, Debug)]
pub enum NormalError {
    #[error("invalid variance")]
    InvalidVariance,
}

impl Normal {
    pub fn new(mean: f64, std_dev: f64) -> Self {
        Self { mean, std_dev }
    }

    pub fn from(mean: f64, var: f64) -> Result<Self, Box<dyn Error>> {
        if var <= 0.0 {
            Err(NormalError::InvalidVariance.into())
        } else {
            Ok(Self::new(mean, var.sqrt()))
        }
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    pub fn std_dev(&self) -> f64 {
        self.std_dev
    }

    pub fn var(&self) -> f64 {
        self.std_dev.powi(2)
    }
}

impl Distribution<f64> for Normal {
    fn p(&self, x: &f64) -> Result<f64, Box<dyn Error>> {
        Ok(1.0 / (2.0 * PI * self.std_dev.powi(2)).sqrt()
            * (-(x - self.mean).powi(2) / (2.0 * self.std_dev.powi(2))).exp())
    }

    fn sample(&self, rng: &mut StdRng) -> Result<f64, Box<dyn Error>> {
        let normal = match RandNormal::new(self.mean, self.std_dev) {
            Ok(n) => n,
            Err(_) => return Err(NormalError::InvalidVariance.into()),
        };

        Ok(rng.sample(normal))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
