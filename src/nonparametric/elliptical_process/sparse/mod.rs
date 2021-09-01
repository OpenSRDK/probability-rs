pub mod regressor;

use super::{BaseEllipticalProcessParams, EllipticalProcessParams};
use crate::nonparametric::{ey, kernel_matrix};
use crate::{opensrdk_linear_algebra::*, RandomVariable};
use crate::{DistributionError, EllipticalParams};
use ey::y_ey;
use opensrdk_kernel_method::*;
pub use rayon::prelude::*;
pub use regressor::*;
use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct SparseEllipticalProcessParams<K, T>
where
    K: Kernel<T>,
    T: RandomVariable,
{
    base: BaseEllipticalProcessParams<K, T>,
    mu: Vec<f64>,
    lkxx: Matrix,
    u: Vec<T>,
    lkuu: Matrix,
    kux: Matrix,
    omega: Vec<f64>,
    ls: Matrix,
    s_inv_kux_omega_y: Matrix,
    mahalanobis_squared: f64,
}

impl<K, T> SparseEllipticalProcessParams<K, T>
where
    K: Kernel<T>,
    T: RandomVariable,
{
    fn new(base: BaseEllipticalProcessParams<K, T>, y: &[f64], u: Vec<T>) -> Self {
        let n = base.x.len();
        let ey = ey(y);
        let mu = vec![ey; n].col_mat();
        let kuu = kernel_matrix(&base.kernel, &base.theta, &u, &u)?;
        let kux = kernel_matrix(&base.kernel, &base.theta, &u, &base.x)?;

        let omega = (0..n)
            .into_par_iter()
            .map(|i| -base.sigma.powi(2))
            .collect::<Vec<_>>();
        let s = &kuu + kux.t() * Matrix::diag(&omega).diinv() * &kux;
        let ls = s.potrf()?;

        let lkuu = kuu.potrf()?;
        let lkxx = kux.t() * lkuu.clone().potri()?.potrf()?;
        let y_ey = y_ey(y, ey).col_mat();
        let s_inv_kux_omega_y = ls.potrs(&kux * omega.clone().col_mat().hadamard_prod(&y_ey))?;

        let mahalanobis_squared = Self::sigma_inv_mul(&kux, &omega, &ls, y_ey);

        Self {
            base,
            mu,
            lkxx,
            u,
            lkuu,
            kux,
            omega,
            ls,
            s_inv_kux_omega_y,
            mahalanobis_squared,
        }
    }

    fn sigma_inv_mul(
        kux: &Matrix,
        omega: &Vec<f64>,
        ls: &Matrix,
        v: Matrix,
    ) -> Result<Matrix, DistributionError> {
        let omega_inv = Matrix::diag(omega).diinv();

        Ok(&omega_inv * v.clone() - &omega_inv * kux.t() * ls.potrs(&kux * &omega_inv * v)?)
    }
}

impl<K, T> BaseEllipticalProcessParams<K, T>
where
    K: Kernel<T>,
    T: RandomVariable,
{
    pub fn sparse(self, y: &[f64], u: Vec<T>) -> SparseEllipticalProcessParams<K, T> {
        SparseEllipticalProcessParams::new(self, y, u)
    }
}

impl<K, T> EllipticalParams for SparseEllipticalProcessParams<K, T>
where
    K: Kernel<T>,
    T: RandomVariable,
{
    fn mu(&self) -> &Vec<f64> {
        &self.mu
    }

    fn sigma_inv_mul(&self, v: Matrix) -> Result<Matrix, DistributionError> {
        Self::sigma_inv_mul(&self.kux, &self.omega, &self.ls, v)
    }

    fn sigma_det_sqrt(&self) -> Result<f64, DistributionError> {
        todo!()
    }

    fn lsigma_cols(&self) -> usize {
        self.lkxx.cols()
    }

    fn sample(&self, z: Vec<f64>) -> Result<Vec<f64>, DistributionError> {
        Ok(self.mu.col_mat() + self.lkxx * z.col_mat())
    }
}

impl<K, T> EllipticalProcessParams<K, T> for SparseEllipticalProcessParams<K, T>
where
    K: Kernel<T>,
    T: RandomVariable,
{
    fn mahalanobis_squared(&self) -> f64 {
        self.mahalanobis_squared
    }
}
