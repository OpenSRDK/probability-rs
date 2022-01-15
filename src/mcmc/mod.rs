pub mod elliptical_slice_sampling;
pub mod hamiltonian;
pub mod importance_sampling;
pub mod metropolis;
pub mod metropolis_hastings;
pub mod sir;
pub mod slice_sampling;

pub use elliptical_slice_sampling::*;
pub use hamiltonian::*;
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

impl<T, U> TransformVec for (T, U)
where
    T: TransformVec,
    U: TransformVec,
{
    type T = (usize, T::T, U::T);

    fn transform_vec(self) -> (Vec<f64>, Self::T) {
        let t = self.0.transform_vec();
        let u = self.1.transform_vec();
        let len = t.0.len();

        ([t.0, u.0].concat(), (len, t.1, u.1))
    }

    fn restore(v: Vec<f64>, info: Self::T) -> Self {
        let (len, t_1, u_1) = info;
        let t_0 = v[0..len].to_vec();
        let u_0 = v[len..].to_vec();

        (T::restore(t_0, t_1), U::restore(u_0, u_1))
    }
}
