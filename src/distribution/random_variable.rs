use opensrdk_linear_algebra::{pp::trf::PPTRF, Matrix, SymmetricPackedMatrix};
use std::fmt::Debug;

use crate::DistributionError;

pub trait RandomVariable: Clone + Debug + Send + Sync {
    type RestoreInfo: Eq;
    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo);
    fn restore(v: &[f64], info: Self::RestoreInfo) -> Result<Self, DistributionError>;
}

impl RandomVariable for () {
    type RestoreInfo = ();

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        (vec![], ())
    }

    fn restore(v: &[f64], info: Self::RestoreInfo) -> Result<Self, DistributionError> {
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

    fn restore(v: &[f64], info: Self::RestoreInfo) -> Result<Self, DistributionError> {
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

    fn restore(v: &[f64], info: Self::RestoreInfo) -> Result<Self, DistributionError> {
        if v.len() != 0 {
            return Err(DistributionError::InvalidRestoreVector);
        }
        Ok(info)
    }
}

impl RandomVariable for usize {
    type RestoreInfo = usize;

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        (vec![], *self)
    }

    fn restore(v: &[f64], info: Self::RestoreInfo) -> Result<Self, DistributionError> {
        if v.len() != 0 {
            return Err(DistributionError::InvalidRestoreVector);
        }
        Ok(info)
    }
}

impl RandomVariable for bool {
    type RestoreInfo = bool;

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        (vec![], *self)
    }

    fn restore(v: &[f64], info: Self::RestoreInfo) -> Result<Self, DistributionError> {
        if v.len() != 0 {
            return Err(DistributionError::InvalidRestoreVector);
        }
        Ok(info)
    }
}

impl RandomVariable for Matrix {
    type RestoreInfo = usize;

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        let rows = self.rows();
        (self.clone().vec(), rows)
    }

    fn restore(v: &[f64], info: Self::RestoreInfo) -> Result<Self, DistributionError> {
        if v.len() != info * info {
            return Err(DistributionError::InvalidRestoreVector);
        }
        Ok(Matrix::from(info, v.to_vec()).unwrap())
    }
}

impl RandomVariable for PPTRF {
    type RestoreInfo = usize;

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        let n = self.0.dim();
        (self.0.elems().to_vec(), n)
    }

    fn restore(v: &[f64], info: Self::RestoreInfo) -> Result<Self, DistributionError> {
        if v.len() != info + (info + 1) / 2 {
            return Err(DistributionError::InvalidRestoreVector);
        }
        Ok(PPTRF(
            SymmetricPackedMatrix::from(info, v.to_vec()).unwrap(),
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

    fn restore(v: &[f64], info: Self::RestoreInfo) -> Result<Self, DistributionError> {
        let (len, t_1, u_1) = info;
        let t_0 = &v[0..len];
        let u_0 = &v[len..];

        Ok((T::restore(t_0, t_1)?, U::restore(u_0, u_1)?))
    }
}

impl<T> RandomVariable for Vec<T>
where
    T: RandomVariable,
{
    type RestoreInfo = Vec<T::RestoreInfo>;

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        let t = self.transform_vec();
        (t.0, t.1)
    }

    fn restore(v: &[f64], info: Self::RestoreInfo) -> Result<Self, DistributionError> {
        if v.len() != 0 {
            return Err(DistributionError::InvalidRestoreVector);
        }
        let t_0 = v.to_vec();
        let t_1 = info;
        // t_1の1要素を2乗してt_0を分解
        let t = T::restore(&t_0[0..t_1[0]], t_1[0])?;
    }
}
