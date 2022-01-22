use crate::{DistributionError, RandomVariable};

#[derive(Clone, Debug)]
pub struct CategoricalParams {
    p: Vec<f64>,
}

impl CategoricalParams {
    pub fn new(p: Vec<f64>) -> Result<Self, DistributionError> {
        Ok(Self { p })
    }

    pub fn p(&self) -> &Vec<f64> {
        &self.p
    }
}

impl RandomVariable for CategoricalParams {
    type RestoreInfo = ();

    fn transform_vec(self) -> (Vec<f64>, Self::RestoreInfo) {
        todo!()
    }

    fn restore(v: Vec<f64>, info: Self::RestoreInfo) -> Self {
        todo!()
    }
}
