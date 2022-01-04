pub mod elliptical_slice_sampling;
pub mod importance_sampling;
pub mod metropolis;
pub mod metropolis_hastings;
pub mod sir;
pub mod slice_sampling;

pub use elliptical_slice_sampling::*;
pub use importance_sampling::*;
pub use metropolis::*;
pub use metropolis_hastings::*;
pub use sir::*;
pub use slice_sampling::*;

use crate::RandomVariable;
use opensrdk_linear_algebra::Matrix;

pub trait TransformVec: RandomVariable {
    type T: Eq;
    fn transform_vec(self) -> (Vec<f64>, Self::T);
    fn restore(v: Vec<f64>, info: Self::T) -> Self;
}

impl TransformVec for f64 {
    type T = ();

    fn transform_vec(self) -> (Vec<f64>, Self::T) {
        (vec![self], ())
    }

    fn restore(v: Vec<f64>, _: Self::T) -> Self {
        v[0]
    }
}

impl TransformVec for Vec<f64> {
    type T = ();

    fn transform_vec(self) -> (Vec<f64>, Self::T) {
        (self, ())
    }

    fn restore(v: Vec<f64>, _: Self::T) -> Self {
        v
    }
}

impl TransformVec for Matrix {
    type T = usize;

    fn transform_vec(self) -> (Vec<f64>, Self::T) {
        let rows = self.rows();
        (self.vec(), rows)
    }

    fn restore(v: Vec<f64>, info: Self::T) -> Self {
        Matrix::from(info, v).unwrap()
    }
}
