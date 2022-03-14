use opensrdk_linear_algebra::{pp::trf::PPTRF, Matrix, SymmetricPackedMatrix};
use std::fmt::Debug;

use crate::DistributionError;

pub trait RandomVariable: Clone + Debug + Send + Sync {
    type RestoreInfo: Eq;
    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo);
    fn len(&self) -> usize;
    fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError>;
}

impl RandomVariable for () {
    type RestoreInfo = ();

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        (vec![], ())
    }

    fn len(&self) -> usize {
        0
    }

    fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        if v.len() != 0 {
            return Err(DistributionError::InvalidRestoreVector);
        }
        Ok(())
    }
}

impl RandomVariable for f64 {
    type RestoreInfo = ();

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        (vec![*self], ())
    }

    fn len(&self) -> usize {
        1
    }

    fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        if v.len() != 1 {
            return Err(DistributionError::InvalidRestoreVector);
        }
        Ok(v[0])
    }
}

impl RandomVariable for u64 {
    type RestoreInfo = u64;

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        (vec![], *self)
    }

    fn len(&self) -> usize {
        1
    }

    fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        if v.len() != 0 {
            return Err(DistributionError::InvalidRestoreVector);
        }
        Ok(*info)
    }
}

impl RandomVariable for usize {
    type RestoreInfo = usize;

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        (vec![], *self)
    }

    fn len(&self) -> usize {
        1
    }

    fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        if v.len() != 0 {
            return Err(DistributionError::InvalidRestoreVector);
        }
        Ok(*info)
    }
}

impl RandomVariable for bool {
    type RestoreInfo = bool;

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        (vec![], *self)
    }

    fn len(&self) -> usize {
        1
    }

    fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        if v.len() != 0 {
            return Err(DistributionError::InvalidRestoreVector);
        }
        Ok(*info)
    }
}

impl RandomVariable for Matrix {
    type RestoreInfo = usize;

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        let rows = self.rows();
        (self.clone().vec(), rows)
    }

    fn len(&self) -> usize {
        self.cols() * self.rows()
    }

    fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        if v.len() != info * info {
            return Err(DistributionError::InvalidRestoreVector);
        }
        Ok(Matrix::from(*info, v.to_vec()).unwrap())
    }
}

impl RandomVariable for PPTRF {
    type RestoreInfo = usize;

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        let n = self.0.dim();
        (self.0.elems().to_vec(), n)
    }

    fn len(&self) -> usize {
        self.0.elems().len()
    }

    fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        if v.len() != info + (info + 1) / 2 {
            return Err(DistributionError::InvalidRestoreVector);
        }
        Ok(PPTRF(
            SymmetricPackedMatrix::from(*info, v.to_vec()).unwrap(),
        ))
    }
}

impl<T, U> RandomVariable for (T, U)
where
    T: RandomVariable,
    U: RandomVariable,
{
    type RestoreInfo = (usize, T::RestoreInfo, U::RestoreInfo);

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        let t = self.0.transform_vec();
        let u = self.1.transform_vec();
        let len = t.0.len();

        ([t.0, u.0].concat(), (len, t.1, u.1))
    }

    fn len(&self) -> usize {
        self.0.len() + self.1.len()
    }

    fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        let (len, t_1, u_1) = info;
        let t_0 = &v[0..*len];
        let u_0 = &v[*len..];

        Ok((T::restore(t_0, t_1)?, U::restore(u_0, u_1)?))
    }
}

impl<T> RandomVariable for Vec<T>
where
    T: RandomVariable,
{
    type RestoreInfo = (Vec<usize>, Vec<T::RestoreInfo>);

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        let len = self.len();
        let mut t_0_vec = vec![];
        let mut len_vec = vec![];
        let mut t_1_vec = vec![];
        for i in 0..len {
            t_0_vec = [t_0_vec, self[i].transform_vec().0].concat();
            len_vec.push(self[i].transform_vec().0.len());
            t_1_vec.push(self[i].transform_vec().1);
        }
        (t_0_vec, (len_vec, t_1_vec))
    }

    fn len(&self) -> usize {
        self.iter().map(|self_i| self_i.len()).sum::<usize>()
    }

    fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        let len_vec = &info.0;
        let t_1_vec = &info.1;
        if len_vec.len() != t_1_vec.len() {
            return Err(DistributionError::InvalidRestoreVector);
        }
        let len = len_vec.len();
        let mut t_vec = vec![];
        let mut n = 0;

        for i in 0..len {
            let len_i = len_vec[i];
            let t_1_i = &t_1_vec[i];
            t_vec.push(T::restore(&v[n..n + len_i], t_1_i)?);
            n += len_vec[i];
        }
        Ok(t_vec)
    }
}
