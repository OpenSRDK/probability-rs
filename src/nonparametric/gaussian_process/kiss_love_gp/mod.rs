pub mod distribution;
pub mod grid;
pub mod internal;

use super::{ey::ey, ey::y_ey, GaussianProcess, GaussianProcessError, GaussianProcessParams};
use crate::opensrdk_linear_algebra::*;
use crate::MultivariateNormalParams;
use grid::Grid;
use opensrdk_kernel_method::{Convolutable, Convolutional, Kernel};
use rayon::prelude::*;
use std::{error::Error, marker::PhantomData};

/// # Lanczos Variance Estimate Kernel Interpolation for Scalable Structured Gaussian Process
/// |                 | time                                                                  |
/// | --------------- | --------------------------------------------------------------------- |
/// | pre-computation | ![tex](https://latex.codecogs.com/svg.latex?O%28KN+KM\log{M}%29)      |
/// | prediction      | ![tex](https://latex.codecogs.com/svg.latex?O%28K%29)                 |
///
/// where ![tex](https://latex.codecogs.com/svg.latex?k=100) here.
pub struct KissLoveGP<K, T>
where
    K: Kernel<Vec<f64>>,
    T: Convolutable + PartialEq,
{
    kernel: Convolutional<K>,
    kernel_params_len: usize,
    theta: Vec<f64>,
    ready_to_predict: bool,
    ey: f64,
    wx: Vec<SparseMatrix>,
    u: Grid,
    a: Vec<Matrix>,
    s: Vec<Matrix>,
    phantom: PhantomData<T>,
}

impl<K, T> KissLoveGP<K, T>
where
    K: Kernel<Vec<f64>>,
    T: Convolutable + PartialEq,
{
    pub fn from(kernel: K) -> Self {
        Self::new(Convolutional::new(kernel))
    }
}

impl<K, T> GaussianProcess<Convolutional<K>, T> for KissLoveGP<K, T>
where
    K: Kernel<Vec<f64>>,
    T: Convolutable + PartialEq,
{
    /// Create an LOVE KISS gp struct.
    ///
    /// | args       |                                                                         |
    /// | ---------- | ----------------------------------------------------------------------- |
    /// | `kernel`   | ![tex](https://latex.codecogs.com/svg.latex?k\in\mathbb{K})             |
    /// | `sigma`    | ![tex](https://latex.codecogs.com/svg.latex?\sigma\in\mathbb{R}_+)      |
    /// | `x`        | ![tex](https://latex.codecogs.com/svg.latex?X\in\mathbb{D}^N)           |
    /// | `y`        | ![tex](https://latex.codecogs.com/svg.latex?\mathbf{y}\in\mathbb{R}^N)  |
    ///
    /// | result |               |
    /// | ------ | ------------- |
    /// | `Ok`   | `Self`        |
    /// | `Err`  | Error message |
    ///
    /// ```
    fn new(kernel: Convolutional<K>) -> Self {
        let params_len = kernel.kernel_ref().params_len();
        Self {
            kernel,
            kernel_params_len: params_len,
            theta: vec![1000.0; params_len],
            ready_to_predict: false,
            ey: 0.0,
            wx: vec![],
            u: Grid::new(vec![]),
            a: vec![],
            s: vec![],
            phantom: PhantomData::<T>,
        }
    }

    fn set_x(&mut self, x: Vec<T>) -> Result<&mut Self, Box<dyn Error>> {
        let (wx, u) = self.wx(&x, false)?;

        self.wx = wx;

        if let Some(u) = u {
            self.u = u;
        }

        Ok(self)
    }

    fn set_theta(&mut self, theta: Vec<f64>) -> Result<&mut Self, Box<dyn Error>> {
        let params_len = self.kernel_params_len;
        if theta.len() != params_len {
            return Err(GaussianProcessError::DimensionMismatch.into());
        }

        self.theta = theta;
        self.reset_prepare();

        Ok(self)
    }

    fn kernel(&self) -> &Convolutional<K> {
        &self.kernel
    }

    fn theta(&self) -> &[f64] {
        &self.theta
    }

    fn n(&self) -> usize {
        if self.wx.len() != 0 {
            return self.wx[0].cols;
        }
        0
    }

    fn ey(&self) -> f64 {
        self.ey
    }

    fn prepare_predict(&mut self, y: &[f64]) -> Result<(), Box<dyn Error>> {
        let n = self.n();
        if n == 0 {
            return Err(GaussianProcessError::Empty.into());
        }

        if n != y.len() {
            return Err(GaussianProcessError::DimensionMismatch.into());
        }

        let m = self.m();
        const K: usize = 100;
        let k = n.min(K);
        let p = self.wx.len();

        self.ey = ey(y);
        let y_ey = &y_ey(y, self.ey);

        let kuu = &self.u.kuu(&self.kernel, &self.theta)?;
        let wx = &self.wx;

        let wxt_kuu_wx_vec_mul = move |v: Vec<f64>| Self::wxt_kuu_wx_vec_mul(&v, wx, kuu);

        let wxt_kuu_wx_inv_y = Matrix::posv_cgm(&wxt_kuu_wx_vec_mul, y_ey.to_vec(), K)?.col_mat();

        self.a = (0..p)
            .into_iter()
            .map(|pi| {
                let wxpi = &wx[pi];

                // a = kuu * wx * (wxt * kuu *wx)^{-1} * y
                let a = kuu.vec_mul((wxpi * &wxt_kuu_wx_inv_y).vec())?.col_mat();
                Ok(a)
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

        // (wxt * kuu * wx)^{-1} = q * t^{-1} * qt
        // q: n×k
        // t: k×k
        let (q, t) = Matrix::sytrd_k(n, k, &wxt_kuu_wx_vec_mul, None)?;

        // t = l * d * lt
        let (l, d) = t.pttrf()?;

        self.s = (0..p)
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
                    .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
                // r = qt * wxt * kuu
                let r = Matrix::from(m, kuu_wx_r_cols.concat());

                // kuu - rt * (l * d * lt)^{-1} * r = q2 * t2 * q2t
                let (q2, t2) = Matrix::sytrd_k(
                    m,
                    100,
                    &|v| {
                        Ok((kuu.vec_mul(v.clone())?.col_mat()
                            - r.t() * l.pttrs(&d, &r * v.col_mat())?.vec().col_mat())
                        .vec())
                    },
                    None,
                )?;

                // t2 = l2 * d2 * l2t
                let (l2, d2) = t2.pttrf()?;
                let d2_sqrt = Matrix::diag(&d2).dipowf(0.5);
                // l2' = l2 * \sqrt{d2}
                // kuu - rt * l * d * lt * r = q2 * l2' * l2't * q2t
                let l2_prime = l2.mat(false) * d2_sqrt;

                // s = q2 * l2'
                let s = q2 * l2_prime;

                Ok(s)
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

        self.ready_to_predict = true;

        Ok(())
    }

    fn predict_multivariate(&self, xs: &[T]) -> Result<MultivariateNormalParams, Box<dyn Error>> {
        if !self.ready_to_predict {
            return Err(GaussianProcessError::NotPrepared.into());
        }
        let len = xs.len();
        if len == 0 {
            return Err(GaussianProcessError::Empty.into());
        }

        let wxs = &self.u.interpolation_weight(xs)?;
        let p = self.a.len();

        if p != wxs.len() {
            return Err(GaussianProcessError::DimensionMismatch.into());
        }

        let (mu, l_sigma) = (0..p)
            .into_iter()
            .map(|pi| {
                let wxspi = &wxs[pi];
                let wxspit = &wxspi.t();

                let api = &self.a[pi];
                let spi = &self.s[pi];

                let mupi = (wxspit * api).vec();
                let l_sigma_pi = wxspit * spi;

                Ok((mupi, l_sigma_pi))
            })
            .try_fold::<(Vec<f64>, Matrix), _, Result<(Vec<f64>, Matrix), Box<dyn Error>>>(
                (vec![self.ey; len], Matrix::new(len, len)),
                |a, b: Result<(Vec<f64>, Matrix), Box<dyn Error>>| {
                    let b = b?;
                    Ok(((a.0.col_mat() + b.0.col_mat()).vec(), a.1 + b.1))
                },
            )?;

        MultivariateNormalParams::new(mu, l_sigma)
    }

    fn kxx_inv_vec(
        &self,
        vec: Vec<f64>,
        params: &GaussianProcessParams<T>,
        with_det_lkxx: bool,
    ) -> Result<(Vec<f64>, Option<f64>), Box<dyn Error>> {
        const K: usize = 100;
        let (wx, kuu) = self.handle_temporal_params(params)?;

        let det = if with_det_lkxx {
            Some(Self::det_kxx(&kuu, &wx)?.sqrt())
        } else {
            None
        };
        let wxt_kuu_wx_vec_mul = move |v: Vec<f64>| Self::wxt_kuu_wx_vec_mul(&v, &wx, &kuu);

        let wxt_kuu_wx_inv_vec = Matrix::posv_cgm(&wxt_kuu_wx_vec_mul, vec, K)?;

        Ok((wxt_kuu_wx_inv_vec, det))
    }

    fn lkxx_vec(
        &self,
        vec: Vec<f64>,
        params: &GaussianProcessParams<T>,
    ) -> Result<Vec<f64>, Box<dyn Error>> {
        let (wx, kuu) = self.handle_temporal_params(params)?;
        let n = self.n();
        let lkuu = Self::lkuu(kuu)?;

        let lkuu_vec = lkuu.vec_mul(vec)?.col_mat();

        let wxt_lkuu_vec = wx
            .par_iter()
            .map(|wxpi| {
                let wxpit = wxpi.t();
                wxpit * &lkuu_vec
            })
            .reduce(|| Matrix::new(n, 1), |a, b| a + b);

        Ok(wxt_lkuu_vec.vec())
    }
}
