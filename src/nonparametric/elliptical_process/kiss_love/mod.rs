pub mod axis;
pub mod grid;
pub mod regressor;

pub use axis::*;
pub use grid::*;
pub use regressor::*;

use rayon::prelude::*;

use super::{BaseEllipticalProcessParams, EllipticalProcessParams};
use crate::nonparametric::{ey, EllipticalProcessError};
use crate::{opensrdk_linear_algebra::*, RandomVariable};
use crate::{DistributionError, EllipticalParams};
use ey::y_ey;
use opensrdk_kernel_method::*;
use std::cmp::Ordering;
use std::error::Error;

const K: usize = 100;

#[derive(Clone, Debug)]
pub struct KissLoveEllipticalProcessParams<K, T>
where
    K: Kernel<Vec<f64>>,
    T: RandomVariable + Convolutable,
{
    base: BaseEllipticalProcessParams<Convolutional<K>, T>,
    mu: Vec<f64>,
    u: Grid,
    a: Vec<Matrix>,
    s: Vec<Matrix>,
    wx: Vec<SparseMatrix>,
    kuu: KroneckerMatrices,
    lkuu: KroneckerMatrices,
    kxx_det_sqrt: f64,
    mahalanobis_squared: f64,
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
        let mu = vec![ey; base.x.len()];

        let (wx, u) = Self::wx_u(&base.x)?;
        let wx = wx;
        let kuu = u.kuu(&base.kernel, &base.theta)?;
        let lkuu = KroneckerMatrices::new(
            kuu.clone()
                .eject()
                .into_iter()
                .map(|kpi| -> Result<_, DistributionError> { Ok(kpi.potrf()?.0) })
                .collect::<Result<Vec<_>, _>>()?,
        );

        let m = kuu.rows();
        let k = n.min(K);
        let p = wx.len();

        let y_ey = y_ey(y, ey).col_mat();

        let sigma2 = base.sigma.powi(2);

        let sigma_inv_y = Self::sigma_inv_mul_with_params(n, sigma2, &wx, &kuu, &y_ey)?;

        let a = (0..p)
            .into_iter()
            .map(|pi| {
                let wxpi = &wx[pi];

                // a = kuu * wx * (wxt * kuu *wx)^{-1} * y
                let a = kuu.vec_mul((wxpi * &sigma_inv_y).vec())?.col_mat();
                Ok(a)
            })
            .collect::<Result<Vec<_>, DistributionError>>()?;

        // (wxt * kuu * wx)^{-1} = q * t^{-1} * qt
        // q: n×k
        // t: k×k
        let (q, t) = Matrix::sytrd_k(n, k, &|v| Self::sigma_mul(n, sigma2, &wx, &kuu, &v), None)?;

        // t = l * d * lt
        let pttrf_result = t.pttrf()?;

        let s = (0..p)
            .into_iter()
            .map(|pi| {
                let wxpi = &wx[pi];
                // let wxpit = &wxpi.t();

                let wx_q = wxpi * &q;

                // rt = kuu * wx * q
                // rt: m,
                let kuu_wx_r_cols = (0..k)
                    .into_iter()
                    .map(|ki| &wx_q[ki])
                    .map(|wx_q_col| Ok(kuu.vec_mul(wx_q_col.to_owned())?))
                    .collect::<Result<Vec<_>, DistributionError>>()?;
                // r = qt * wxt * kuu
                let r = Matrix::from(m, kuu_wx_r_cols.concat())?;

                // kuu - rt * (l * d * lt)^{-1} * r = q2 * t2 * q2t
                let (q2, t2) = Matrix::sytrd_k(
                    m,
                    100,
                    &|v| {
                        Ok((kuu.vec_mul(v.clone())?.col_mat()
                            - r.t() * pttrf_result.pttrs(&r * v.col_mat())?.vec().col_mat())
                        .vec())
                    },
                    None,
                )?;

                // t2 = l2 * d2 * l2t
                let pttrf_result2 = t2.pttrf()?;
                let d2_sqrt = pttrf_result2.1.powf(0.5);
                // l2' = l2 * \sqrt{d2}
                // kuu - rt * l * d * lt * r = q2 * l2' * l2't * q2t
                let l2_prime = pttrf_result2.0.mat(false) * d2_sqrt.mat();

                // s = q2 * l2'
                let s = q2 * l2_prime;

                Ok(s)
            })
            .collect::<Result<Vec<_>, DistributionError>>()?;

        let kxx_det_sqrt = Self::det_kxx(&wx, &kuu, sigma2)?.sqrt();
        let mahalanobis_squared = (y_ey.t() * &sigma_inv_y)[(0, 0)];

        Ok(Self {
            base,
            mu,
            u,
            a,
            s,
            wx,
            kuu,
            lkuu,
            kxx_det_sqrt,
            mahalanobis_squared,
        })
    }

    fn wx_u(x: &Vec<T>) -> Result<(Vec<SparseMatrix>, Grid), DistributionError> {
        let n = x.len();

        let parts_len = x[0].parts_len();
        let data_len = x[0].data_len();
        if parts_len == 0 || data_len == 0 {
            return Err(DistributionError::InvalidParameters(
                EllipticalProcessError::Empty.into(),
            ));
        }

        let points = vec![(n / 2usize.pow(data_len as u32)).max(2); data_len];
        let u = Grid::from(&x, &points)?;
        let wx = u.interpolation_weight(&x)?;

        return Ok((wx, u));
    }

    fn wxt_kuu_wx_mul(
        v: &Vec<f64>,
        wx: &Vec<SparseMatrix>,
        kuu: &KroneckerMatrices,
    ) -> Result<Vec<f64>, DistributionError> {
        wx.iter()
            .map(|wxpi| {
                let v = v.clone().col_mat();
                let wx_v = wxpi * &v;
                let kuu_wx_v = kuu.vec_mul(wx_v.vec())?.col_mat();
                let wxt_kuu_wx_v = wxpi.t() * kuu_wx_v;
                Ok(wxt_kuu_wx_v.vec())
            })
            .try_fold(vec![0.0; v.len()], |a, b: Result<_, DistributionError>| {
                Ok((a.col_mat() + b?.col_mat()).vec())
            })
    }

    /// See Andrew Gordon Wilson
    fn det_kxx(
        wx: &Vec<SparseMatrix>,
        kuu: &KroneckerMatrices,
        sigma2: f64,
    ) -> Result<f64, DistributionError> {
        let m = wx[0].rows;
        let n = wx[0].cols;

        let kuu_toeplitz = kuu
            .matrices()
            .iter()
            .map(|kp| Ok(ToeplitzMatrix::from(kp[0].to_vec(), kp[0][1..].to_vec())?))
            .collect::<Result<Vec<_>, DistributionError>>()?;

        let lambda = kuu_toeplitz
            .par_iter()
            .map(|kp| {
                kp.embedded_circulant().cievd().1[..kp.dim()]
                    .to_owned()
                    .col_mat()
            })
            .collect::<Vec<_>>();
        let lambda = KroneckerMatrices::new(lambda);
        let mut lambda = lambda.prod().vec();

        lambda.sort_by(|a, b| {
            a.re.partial_cmp(&b.re).unwrap_or(if !a.re.is_finite() {
                Ordering::Less
            } else {
                Ordering::Greater
            })
        });
        if !lambda[0].re.is_finite() {
            return Err(DistributionError::Others(
                EllipticalProcessError::NaNContamination.into(),
            ));
        }

        let lambda = &lambda[m - n..];

        let det = lambda
            .par_iter()
            .map(|lmd| ((n as f64 / m as f64) * lmd.re + sigma2))
            .product::<f64>();

        Ok(det)
    }

    fn sigma_mul(
        n: usize,
        sigma2: f64,
        wx: &Vec<SparseMatrix>,
        kuu: &KroneckerMatrices,
        v: &Vec<f64>,
    ) -> Result<Vec<f64>, Box<dyn Error + Send + Sync>> {
        match Self::wxt_kuu_wx_mul(v, wx, kuu) {
            Ok(v) => Ok((v.clone().col_mat() + (vec![sigma2; n].diag() * v).col_mat()).vec()),
            Err(e) => Err(e.into()),
        }
    }

    fn sigma_inv_mul_with_params(
        n: usize,
        sigma2: f64,
        wx: &Vec<SparseMatrix>,
        kuu: &KroneckerMatrices,
        v: &Matrix,
    ) -> Result<Matrix, DistributionError> {
        let sigma_mul = move |ve: Vec<f64>| Self::sigma_mul(n, sigma2, wx, kuu, &ve);

        let sigma_inv_v = Matrix::from(
            v.rows(),
            (0..v.cols())
                .into_par_iter()
                .map(|i| Matrix::posv_cgm(&sigma_mul, v[i].to_vec(), K))
                .collect::<Result<Vec<_>, _>>()?
                .concat(),
        );

        Ok(sigma_inv_v?)
    }
}

impl<K, T> BaseEllipticalProcessParams<Convolutional<K>, T>
where
    K: Kernel<Vec<f64>>,
    T: RandomVariable + Convolutable,
{
    /// Lanczos Variance Estimate Kernel Interpolation for Scalable Structured Gaussian Process
    ///
    /// - Pre-computation time: O(k(n + m log m))
    /// - Pre-computation storage: O(km)
    /// - Prediction time: O(k)
    /// with k = 100
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
        let n = self.mu().len();
        let sigma2 = self.base.sigma.powi(2);

        Self::sigma_inv_mul_with_params(n, sigma2, &self.wx, &self.kuu, &v)
    }

    fn lsigma_cols(&self) -> usize {
        self.mu.len()
    }

    fn sample(&self, z: Vec<f64>) -> Result<Vec<f64>, DistributionError> {
        let n = self.mu().len();
        let lkuu_z = self.lkuu.vec_mul(z)?.col_mat();
        let wxt_lkuu_z = self
            .wx
            .par_iter()
            .map(|wxpi| {
                let wxpit = wxpi.t();
                wxpit * &lkuu_z
            })
            .reduce(|| Matrix::new(n, 1), |a, b| a + b);

        Ok((self.mu[0] + wxt_lkuu_z).vec())
    }
}

impl<K, T> EllipticalProcessParams<Convolutional<K>, T> for KissLoveEllipticalProcessParams<K, T>
where
    K: Kernel<Vec<f64>>,
    T: RandomVariable + Convolutable,
{
    fn mahalanobis_squared(&self) -> f64 {
        self.mahalanobis_squared
    }
}
