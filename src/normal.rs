use crate::Distribution;
use rand::prelude::*;
use rand_distr::Normal as RandNormal;

pub struct Normal {
    mean: f64,
    var: f64,
}

impl Normal {
    fn new(mean: f64, var: f64) -> Self {
        Self { mean, var }
    }

    pub fn from(mean: f64, var: f64) -> Result<Self, String> {
        if var <= 0.0 {
            Err("variance must be greater than zero".to_owned())
        } else {
            Ok(Self::new(mean, var))
        }
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    pub fn var(&self) -> f64 {
        self.var
    }
}

impl Distribution for Normal {
    fn sample(&self, rng: &mut StdRng) -> Result<f64, String> {
        let normal = match RandNormal::new(self.mean, self.var.sqrt()) {
            Ok(n) => n,
            Err(_) => {
                return Err("too small variance".to_owned());
            }
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
