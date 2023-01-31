use super::AcquisitionFunctions;

pub struct ExpectedImprovement;

impl AcquisitionFunctions for ExpectedImprovement {
    fn value(&self, theta: &crate::NormalParams) -> f64 {
        todo!()
    }
}
