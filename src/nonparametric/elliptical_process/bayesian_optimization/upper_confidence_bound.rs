use super::AcquisitionFunctions;

pub struct UpperConfidenceBound;

impl AcquisitionFunctions for UpperConfidenceBound {
    fn value(&self, theta: &crate::NormalParams) -> f64 {
        todo!()
    }
}
