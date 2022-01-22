use crate::{DistributionError, GammaError, RandomVariable};

#[derive(Clone, Debug, PartialEq)]
pub struct GammaParams {
    shape: f64,
    scale: f64,
}

impl GammaParams {
    pub fn new(shape: f64, scale: f64) -> Result<Self, DistributionError> {
        if shape <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                GammaError::ShapeMustBePositive.into(),
            ));
        }
        if scale <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                GammaError::ScaleMustBePositive.into(),
            ));
        }

        Ok(Self { shape, scale })
    }

    pub fn shape(&self) -> f64 {
        self.shape
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl RandomVariable for GammaParams {
    type RestoreInfo = ();

    fn transform_vec(self) -> (Vec<f64>, Self::RestoreInfo) {
        todo!()
    }

    fn restore(v: Vec<f64>, info: Self::RestoreInfo) -> Self {
        todo!()
    }
}
