use super::grid::InducingGridError;
use crate::DistributionError;

#[derive(Clone, Debug)]
pub struct Axis {
    min: f64,
    max: f64,
    points: usize,
}

impl Axis {
    pub fn new(min: f64, max: f64, points: usize) -> Result<Self, DistributionError> {
        if max <= min {
            return Err(DistributionError::InvalidParameters(
                InducingGridError::InvalidRange.into(),
            ));
        }
        if points < 2 {
            return Err(DistributionError::InvalidParameters(
                InducingGridError::TooLessPoints.into(),
            ));
        }

        Ok(Self { min, max, points })
    }

    pub fn min(&self) -> f64 {
        self.min
    }

    pub fn max(&self) -> f64 {
        self.max
    }

    pub fn points(&self) -> usize {
        self.points
    }

    pub fn value(&self, index: usize) -> f64 {
        self.min + index as f64 * (self.max - self.min) / (self.points - 1) as f64
    }

    pub fn index(&self, value: f64) -> usize {
        ((value - self.min) / (self.max - self.min) * (self.points - 1) as f64) as usize
    }
}

#[cfg(test)]
mod tests {
    use crate::nonparametric::Axis;

    #[test]
    fn it_works() {
        let axis_t = Axis::new(0.0, 10.0, 6).unwrap();
        let value_a = axis_t.value(3);
        let index_t = axis_t.index(value_a);
        let value_b = axis_t.value(index_t);
        assert_eq!(value_a, value_b)
    }

    #[test]
    fn it_works_2() {
        let axis_t = Axis::new(0.0, 10.0, 6).unwrap();
        let index_a = axis_t.index(5.0);
        let value_t = axis_t.value(index_a);
        let index_b = axis_t.index(value_t);
        assert_eq!(index_a, index_b)
    }
}
