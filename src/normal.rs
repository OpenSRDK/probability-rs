use crate::distribution::Distribution;
use rand::prelude::*;
use rand_distr::Normal as RandNormal;

pub struct Normal {
    pub mean: f64,
    pub variance: f64,
}

impl Normal {
    pub fn new(mean: f64, variance: f64) -> Self {
        Self { mean, variance }
    }
}

impl Distribution for Normal {
    fn sample(&self, thread_rng: &mut ThreadRng) -> f64 {
        let normal = RandNormal::new(self.mean, self.variance.sqrt()).unwrap();
        thread_rng.sample(normal)
    }
}
