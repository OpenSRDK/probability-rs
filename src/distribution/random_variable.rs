use std::fmt::Debug;

use opensrdk_linear_algebra::Matrix;

pub trait RandomVariable: Clone + Debug + Send + Sync {
    type RestoreInfo: Eq;
    fn transform_vec(self) -> (Vec<f64>, Self::RestoreInfo);
    fn restore(v: Vec<f64>, info: Self::RestoreInfo) -> Self;
}

impl RandomVariable for () {
    type RestoreInfo = ();

    fn transform_vec(self) -> (Vec<f64>, Self::RestoreInfo) {
        (vec![], ())
    }

    fn restore(v: Vec<f64>, info: Self::RestoreInfo) -> Self {
        ()
    }
}

impl RandomVariable for f64 {
    type RestoreInfo = ();

    fn transform_vec(self) -> (Vec<f64>, Self::RestoreInfo) {
        (vec![self], ())
    }

    fn restore(v: Vec<f64>, info: Self::RestoreInfo) -> Self {
        v[0]
    }
}

impl RandomVariable for u64 {
    type RestoreInfo = ();

    fn transform_vec(self) -> (Vec<f64>, Self::RestoreInfo) {
        todo!()
    }

    fn restore(v: Vec<f64>, info: Self::RestoreInfo) -> Self {
        todo!()
    }
}

impl RandomVariable for usize {
    type RestoreInfo = ();

    fn transform_vec(self) -> (Vec<f64>, Self::RestoreInfo) {
        todo!()
    }

    fn restore(v: Vec<f64>, info: Self::RestoreInfo) -> Self {
        todo!()
    }
}

impl RandomVariable for bool {
    type RestoreInfo = ();

    fn transform_vec(self) -> (Vec<f64>, Self::RestoreInfo) {
        todo!()
    }

    fn restore(v: Vec<f64>, info: Self::RestoreInfo) -> Self {
        todo!()
    }
}

impl RandomVariable for Matrix {
    type RestoreInfo = usize;

    fn transform_vec(self) -> (Vec<f64>, Self::RestoreInfo) {
        let rows = self.rows();
        (self.vec(), rows)
    }

    fn restore(v: Vec<f64>, info: Self::RestoreInfo) -> Self {
        Matrix::from(info, v).unwrap()
    }
}

impl<T, U> RandomVariable for (T, U)
where
    T: RandomVariable,
    U: RandomVariable,
{
    type RestoreInfo = (usize, T::RestoreInfo, U::RestoreInfo);

    fn transform_vec(self) -> (Vec<f64>, Self::RestoreInfo) {
        let t = self.0.transform_vec();
        let u = self.1.transform_vec();
        let len = t.0.len();

        ([t.0, u.0].concat(), (len, t.1, u.1))
    }

    fn restore(v: Vec<f64>, info: Self::RestoreInfo) -> Self {
        let (len, t_1, u_1) = info;
        let t_0 = v[0..len].to_vec();
        let u_0 = v[len..].to_vec();

        (T::restore(t_0, t_1), U::restore(u_0, u_1))
    }
}

impl<T> RandomVariable for Vec<T>
where
    T: RandomVariable,
{
    type RestoreInfo = Vec<T::RestoreInfo>;

    fn transform_vec(self) -> (Vec<f64>, Self::RestoreInfo) {
        todo!()
    }

    fn restore(v: Vec<f64>, info: Self::RestoreInfo) -> Self {
        todo!()
    }
}
