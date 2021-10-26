pub mod elliptical_slice;

pub use elliptical_slice::*;

use crate::RandomVariable;
use opensrdk_linear_algebra::Matrix;

pub trait VectorSampleable: RandomVariable {
    type T;
    fn transform_vec(self) -> (Vec<f64>, Self::T);
    fn restore(v: (Vec<f64>, Self::T)) -> Self;
}

impl VectorSampleable for f64 {
    type T = ();

    fn transform_vec(self) -> (Vec<f64>, Self::T) {
        (vec![self], ())
    }

    fn restore(v: (Vec<f64>, Self::T)) -> Self {
        v.0[0]
    }
}

impl VectorSampleable for Vec<f64> {
    type T = ();

    fn transform_vec(self) -> (Vec<f64>, Self::T) {
        (self, ())
    }

    fn restore(v: (Vec<f64>, Self::T)) -> Self {
        v.0
    }
}

impl VectorSampleable for Matrix {
    type T = usize;

    fn transform_vec(self) -> (Vec<f64>, Self::T) {
        let rows = self.rows();
        (self.vec(), rows)
    }

    fn restore(v: (Vec<f64>, Self::T)) -> Self {
        Matrix::from(v.1, v.0)
    }
}
