use crate::Distribution;
use rand::prelude::*;
use rand_distr::Normal as RandNormal;

pub struct Normal {
    mean: f64,
    variance: f64,
}

impl Normal {
    fn new(mean: f64, variance: f64) -> Self {
        Self { mean, variance }
    }

    pub fn from(mean: f64, variance: f64) -> Result<Self, String> {
        if variance <= 0.0 {
            Err("variance must be greater than zero".to_owned())
        } else {
            Ok(Self::new(mean, variance))
        }
    }

    pub fn get_mean(&self) -> f64 {
        self.mean
    }

    pub fn get_variance(&self) -> f64 {
        self.variance
    }
}

impl Distribution for Normal {
    fn sample(&self, thread_rng: &mut ThreadRng) -> Result<f64, String> {
        let normal = match RandNormal::new(self.mean, self.variance.sqrt()) {
            Ok(n) => n,
            Err(_) => {
                return Err("too small variance".to_owned());
            }
        };

        Ok(thread_rng.sample(normal))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
