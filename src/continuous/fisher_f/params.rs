use crate::{DistributionError, FisherFError, RandomVariable};

#[derive(Clone, Debug, PartialEq)]
pub struct FisherFParams {
    m: f64,
    n: f64,
}

impl FisherFParams {
    pub fn new(m: f64, n: f64) -> Result<Self, DistributionError> {
        if m <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                FisherFError::MMustBePositive.into(),
            ));
        }
        if n <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                FisherFError::NMustBePositive.into(),
            ));
        }

        Ok(Self { m, n })
    }

    pub fn m(&self) -> f64 {
        self.m
    }

    pub fn n(&self) -> f64 {
        self.n
    }
}

impl RandomVariable for FisherFParams {
    type RestoreInfo = ();

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        (vec![self.m, self.n], ())
    }

    fn len(&self) -> usize {
        2usize
    }

    fn restore(v: &[f64], _: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        Self::new(v[0], v[1])
    }
}
