pub mod regressor;

pub use rayon::prelude::*;
pub use regressor::*;

use super::{BaseEllipticalProcessParams, EllipticalProcessParams};
use crate::nonparametric::{ey, kernel_matrix, EllipticalProcessError};
use crate::{opensrdk_linear_algebra::*, RandomVariable};
use crate::{DistributionError, EllipticalParams};
use ey::y_ey;
use opensrdk_kernel_method::*;

const K: usize = 100;

#[derive(Clone, Debug)]
pub struct SparseEllipticalProcessParams<K, T>
where
    K: Kernel<T>,
    T: RandomVariable,
{
    base: BaseEllipticalProcessParams<K, T>,
    mu: Vec<f64>,
    lsigma: Matrix,
    u: Vec<T>,
    lkuu: Matrix,
    kux: Matrix,
    omega_inv: DiagonalMatrix,
    ls: Matrix,
    s_inv_kux_omega_y: Matrix,
    kxx_det_sqrt: f64,
    mahalanobis_squared: f64,
}

impl<K, T> SparseEllipticalProcessParams<K, T>
where
    K: Kernel<T>,
    T: RandomVariable,
{
    fn new(
        mut base: BaseEllipticalProcessParams<K, T>,
        y: &[f64],
        u: Vec<T>,
    ) -> Result<Self, DistributionError> {
        let n = y.len();
        if n == 0 {
            return Err(DistributionError::InvalidParameters(
                EllipticalProcessError::Empty.into(),
            ));
        }
        if n != base.x.len() {
            return Err(DistributionError::InvalidParameters(
                EllipticalProcessError::DimensionMismatch.into(),
            ));
        }

        let ey = ey(y);
        let mu = vec![ey; n];
        let kuu = kernel_matrix(&base.kernel, &base.theta, &u, &u)?;
        let kux = kernel_matrix(&base.kernel, &base.theta, &u, &base.x)?;

        let lkuu = kuu.clone().potrf()?;

        let omega = (0..n)
            .into_par_iter()
            .map(|i| {
                (
                    base.kernel
                        .value(&base.theta, &base.x[i], &base.x[i])
                        .unwrap(),
                    kux[i].to_vec().col_mat(),
                )
            })
            .map(|(kxixi, kuxi)| {
                Ok(kxixi - (kuxi.t() * lkuu.potrs(kuxi)?)[0][0] + base.sigma.powi(2))
            })
            .collect::<Result<Vec<_>, MatrixError>>()?
            .diag();
        let y_ey = y_ey(y, ey).col_mat();
        let omega_y = &omega * y_ey.slice().to_vec();

        let omega_inv = omega.diinv();
        let omega_inv_mat = omega_inv.mat();
        let s = &kuu + &kux * &omega_inv_mat * kux.t();
        let ls = s.potrf()?;

        let kux_ref = &kux;
        let omega_inv_ref = &omega_inv_mat;
        let ls_ref = &ls;
        let sigma_inv_mul = move |v: Vec<f64>| match Self::sigma_inv_mul(
            kux_ref,
            omega_inv_ref,
            ls_ref,
            v.col_mat(),
        ) {
            Ok(v) => Ok(v.vec()),
            Err(e) => Err(e.into()),
        };

        let k = n.min(K);
        let (q, t) = Matrix::sytrd_k(n, k, &sigma_inv_mul, None)?;
        let (bd, d) = t.pttrf()?;
        let lkxx = q * (bd.mat(false) * d.mat());

        let s_inv_kux_omega_y = ls.potrs(&kux * omega_y.col_mat())?;

        let kxx_det_sqrt = 0.0; // todo
        let mahalanobis_squared =
            (y_ey.t() * Self::sigma_inv_mul(&kux, omega_inv_ref, &ls, y_ey)?)[0][0];

        base.x = vec![];

        Ok(Self {
            base,
            mu,
            lsigma: lkxx,
            u,
            lkuu,
            kux,
            omega_inv,
            ls,
            kxx_det_sqrt,
            s_inv_kux_omega_y,
            mahalanobis_squared,
        })
    }

    fn sigma_inv_mul(
        kux: &Matrix,
        omega_inv: &Matrix,
        ls: &Matrix,
        v: Matrix,
    ) -> Result<Matrix, DistributionError> {
        Ok(omega_inv * v.clone() - omega_inv * kux.t() * ls.potrs(kux * omega_inv * v)?)
    }
}

impl<K, T> BaseEllipticalProcessParams<K, T>
where
    K: Kernel<T>,
    T: RandomVariable,
{
    /// Elliptical Process with approximation called the Fully Independent Training Conditional (FITC) for scalability.
    ///
    /// - Pre-computation time: O(nm^2)
    /// - Pre-computation storage: O(m^2)
    /// - Prediction time: O(m^2)
    pub fn sparse(
        self,
        y: &[f64],
        u: Vec<T>,
    ) -> Result<SparseEllipticalProcessParams<K, T>, DistributionError> {
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
        Self::sigma_inv_mul(&self.kux, &self.omega_inv.mat(), &self.ls, v)
    }

    fn sigma_det_sqrt(&self) -> f64 {
        self.kxx_det_sqrt
    }

    fn lsigma_cols(&self) -> usize {
        self.lsigma.cols()
    }

    fn sample(&self, z: Vec<f64>) -> Result<Vec<f64>, DistributionError> {
        Ok((self.mu[0] + &self.lsigma * z.col_mat()).vec())
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
