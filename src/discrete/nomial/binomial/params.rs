use crate::{BinominalError, DistributionError, RandomVariable};

#[derive(Clone, Debug, PartialEq)]
pub struct BinomialParams {
    n: u64,
    p: f64,
}

impl BinomialParams {
    pub fn new(n: u64, p: f64) -> Result<Self, DistributionError> {
        if p < 0.0 || 1.0 < p {
            return Err(DistributionError::InvalidParameters(
                BinominalError::PMustBeProbability.into(),
            ));
        }

        Ok(Self { n, p })
    }

    pub fn n(&self) -> u64 {
        self.n
    }

    pub fn p(&self) -> f64 {
        self.p
    }
}

impl RandomVariable for BinomialParams {
    type RestoreInfo = ();

    fn transform_vec(self) -> (Vec<f64>, Self::RestoreInfo) {
        todo!()
    }

    fn restore(v: Vec<f64>, info: Self::RestoreInfo) -> Self {
        todo!()
    }
}
