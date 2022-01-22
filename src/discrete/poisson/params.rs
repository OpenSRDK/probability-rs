use crate::{PoissonError, RandomVariable};

#[derive(Clone, Debug, PartialEq)]
pub struct PoissonParams {
    lambda: f64,
}

impl PoissonParams {
    pub fn new(lambda: f64) -> Result<Self, PoissonError> {
        if lambda <= 0.0 {
            return Err(PoissonError::LambdaMustBePositive.into());
        }

        Ok(Self { lambda })
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl RandomVariable for PoissonParams {
    type RestoreInfo = ();

    fn transform_vec(self) -> (Vec<f64>, Self::RestoreInfo) {
        todo!()
    }

    fn restore(v: Vec<f64>, info: Self::RestoreInfo) -> Self {
        todo!()
    }
}
