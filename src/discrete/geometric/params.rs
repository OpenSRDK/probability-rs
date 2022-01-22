use crate::{GeometricError, RandomVariable};

#[derive(Clone, Debug, PartialEq)]
pub struct GeometricParams {
    p: f64,
}

impl GeometricParams {
    pub fn new(p: f64) -> Result<Self, GeometricError> {
        if p < 0.0 || 1.0 < p {
            return Err(GeometricError::PMustBeProbability.into());
        }

        Ok(Self { p })
    }

    pub fn p(&self) -> f64 {
        self.p
    }
}

impl RandomVariable for GeometricParams {
    type RestoreInfo = ();

    fn transform_vec(self) -> (Vec<f64>, Self::RestoreInfo) {
        todo!()
    }

    fn restore(v: Vec<f64>, info: Self::RestoreInfo) -> Self {
        todo!()
    }
}
