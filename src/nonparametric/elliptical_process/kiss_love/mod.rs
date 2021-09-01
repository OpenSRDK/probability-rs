pub mod grid;
pub mod regressor;

pub use grid::*;
pub use rayon::prelude::*;
pub use regressor::*;

use super::{BaseEllipticalProcessParams, EllipticalProcessParams};
use crate::nonparametric::{ey, kernel_matrix};
use crate::{opensrdk_linear_algebra::*, RandomVariable};
use crate::{DistributionError, EllipticalParams};
use ey::y_ey;
use opensrdk_kernel_method::*;
use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct KissLoveEllipticalProcessParams<K, T>
where
    K: Kernel<Vec<f64>>,
    T: RandomVariable + Convolutable,
{
    base: BaseEllipticalProcessParams<Convolutional<K>, T>,
    mu: Vec<f64>,
    kxx_det_sqrt: f64,
    y_ey: Vec<f64>,
    kxx_inv_y: Matrix,
}

impl<K, T> KissLoveEllipticalProcessParams<K, T>
where
    K: Kernel<Vec<f64>>,
    T: RandomVariable + Convolutable,
{
    fn new(
        base: BaseEllipticalProcessParams<Convolutional<K>, T>,
        y: &[f64],
    ) -> Result<Self, DistributionError> {
        let ey = ey(y);
        let mu = vec![ey; base.x.len()].col_mat();
        let kxx = kernel_matrix(&base.kernel, &base.theta, &base.x, &base.x)?;
        let lkxx = kxx.potrf()?;
        let y_ey = y_ey(y, ey);
        let kxx_inv_y = lkxx.potrs(y_ey.clone().col_mat())?;

        Ok(Self {
            base,
            mu,
            lkxx,
            y_ey,
            kxx_inv_y,
        })
    }
}

impl<K, T> BaseEllipticalProcessParams<Convolutional<K>, T>
where
    K: Kernel<Vec<f64>>,
    T: RandomVariable + Convolutable,
{
    pub fn kiss_love(
        self,
        y: &[f64],
    ) -> Result<KissLoveEllipticalProcessParams<K, T>, DistributionError> {
        KissLoveEllipticalProcessParams::new(self, y)
    }
}

impl<K, T> EllipticalParams for KissLoveEllipticalProcessParams<K, T>
where
    K: Kernel<Vec<f64>>,
    T: RandomVariable + Convolutable,
{
    fn mu(&self) -> &Vec<f64> {
        &self.mu
    }

    fn sigma_inv_mul(&self, v: Matrix) -> Result<Matrix, DistributionError> {
        Ok(self.lkxx.potrs(v)?)
    }

    fn sigma_det_sqrt(&self) -> Result<f64, DistributionError> {
        Ok(self.lkxx.trdet())
    }

    fn lsigma_cols(&self) -> usize {
        self.lkxx.cols()
    }

    fn sample(&self, z: Vec<f64>) -> Result<Vec<f64>, DistributionError> {
        Ok(self.mu.col_mat() + self.lkxx * z.col_mat())
    }
}

impl<K, T> EllipticalProcessParams<Convolutional<K>, T> for KissLoveEllipticalProcessParams<K, T>
where
    K: Kernel<Vec<f64>>,
    T: RandomVariable + Convolutable,
{
    fn mahalanobis_squared(&self) -> f64 {}
}
