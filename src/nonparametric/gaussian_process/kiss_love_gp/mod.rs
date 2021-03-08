pub mod grid;
pub mod internal;

use super::{ey::ey, ey::y_ey, GaussianProcess, GaussianProcessError, GaussianProcessParams};
use crate::MultivariateNormalParams;
use crate::{opensrdk_linear_algebra::*, Distribution};
use grid::Grid;
use opensrdk_kernel_method::{Convolutable, Convolutional, Kernel};
use rand::Rng;
use rand_distr::StandardNormal;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{error::Error, f64::consts::PI, marker::PhantomData};

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

    fn set_x(&mut self, x: Vec<T>) -> Result<&Self, Box<dyn Error>> {
        let (wx, u) = self.wx(&x, false)?;

        self.wx = wx;
        if let Some(u) = u {
            self.u = u;
        }

        Ok(self)
    }

    fn set_theta(&mut self, theta: Vec<f64>) -> Result<&Self, Box<dyn Error>> {
        let params_len = self.kernel_params_len;
        if theta.len() != params_len {
            return Err(GaussianProcessError::DimensionMismatch.into());
        }

        self.theta = theta;
        self.reset_prepare()?;

        Ok(self)
    }

    fn kernel(&self) -> &Convolutional<K> {
        &self.kernel
    }

    fn theta(&self) -> &[f64] {
        &self.theta
    }

    fn prepare_predict(&mut self, y: &[f64]) -> Result<(), Box<dyn Error>> {
        let m = self.wx[0].rows;
        let n = self.wx[0].cols;
        let k = n.min(100);
        let p = self.wx.len();

        self.ey = ey(y);
        let y_ey = &y_ey(y, self.ey);

        let kuu = &self.u.kuu(&self.kernel, &self.theta)?;

        let wx = &self.wx;

        let wxt_kuu_wx_vec_mul = move |v: Vec<f64>| {
            wx.iter()
                .map(|wxpi| {
                    let v = v.clone().col_mat();
                    let wx_v = wxpi * &v;
                    let kzz_wx_v = kuu.vec_mul(wx_v.vec())?.col_mat();
                    let wxt_kzz_wx_v = wxpi.t() * kzz_wx_v;
                    Ok(wxt_kzz_wx_v.vec())
                })
                .try_fold(vec![0.0; v.len()], |a, b: Result<_, Box<dyn Error>>| {
                    Ok((a.col_mat() + b?.col_mat()).vec())
                })
        };

        let wxt_kuu_wx_inv_y = Matrix::posv_cgm(&wxt_kuu_wx_vec_mul, y_ey.to_vec(), k)?.col_mat();

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
}

impl<K, T> Distribution for KissLoveGP<K, T>
where
    K: Kernel<Vec<f64>>,
    T: Convolutable + PartialEq,
{
    type T = Vec<f64>;
    type U = GaussianProcessParams<T>;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, Box<dyn Error>> {
        let y = x;
        let (kuu, wx) = self.for_multivariate_normal(theta)?;
        let n = wx[0].cols;
        let k = n.min(100);

        if y.len() != n {
            return Err(GaussianProcessError::DimensionMismatch.into());
        }

        let det = Self::det_kxx(&kuu, &wx)?;
        let y_ey = y_ey(y, self.ey).col_mat();

        let wxt_kuu_wx_vec_mul = move |v: Vec<f64>| {
            wx.iter()
                .map(|wxpi| {
                    let v = v.clone().col_mat();
                    let wx_v = wxpi * &v;
                    let kzz_wx_v = kuu.vec_mul(wx_v.vec())?.col_mat();
                    let wxt_kzz_wx_v = wxpi.t() * kzz_wx_v;
                    Ok(wxt_kzz_wx_v.vec())
                })
                .try_fold(vec![0.0; v.len()], |a, b: Result<_, Box<dyn Error>>| {
                    Ok((a.col_mat() + b?.col_mat()).vec())
                })
        };

        Ok(1.0 / ((2.0 * PI).powf(n as f64 / 2.0) * det)
            * (-1.0 / 2.0
                * (y_ey.t() * Matrix::posv_cgm(&wxt_kuu_wx_vec_mul, y_ey.vec(), k)?.col_mat())[0]
                    [0])
            .exp())
    }

    fn sample(
        &self,
        theta: &Self::U,
        rng: &mut rand::prelude::StdRng,
    ) -> Result<Self::T, Box<dyn Error>> {
        let (kuu, wx) = self.for_multivariate_normal(theta)?;
        let n = wx[0].cols;
        let luu = Self::luu(kuu)?;

        let z = (0..n)
            .into_iter()
            .map(|_| rng.sample(StandardNormal))
            .collect::<Vec<_>>();
        let luu_z = luu.vec_mul(z)?.col_mat();

        let wxt_luu_z = wx
            .par_iter()
            .map(|wxpi| {
                let wxpit = wxpi.t();
                wxpit * &luu_z
            })
            .reduce(|| Matrix::new(n, 1), |a, b| a + b);

        let mu = vec![self.ey; n].col_mat();
        let y = mu + wxt_luu_z;

        Ok(y.vec())
    }
}
