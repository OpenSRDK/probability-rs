pub mod grid;

use super::{
    ey::ey, ey::y_ey, kernel_matrix::kernel_matrix, GaussianProcess, GaussianProcessError,
    GaussianProcessParams,
};
use crate::MultivariateNormalParams;
use crate::{opensrdk_linear_algebra::*, Distribution};
use grid::Grid;
use opensrdk_kernel_method::{Convolutable, Convolutional, Kernel};
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
    T: Convolutable,
{
    kernel: Convolutional<K>,
    kernel_params_len: usize,
    theta: Vec<f64>,
    ready_to_predict: bool,
    ey: f64,
    wx: Vec<SparseMatrix>,
    u: Grid,
    a: Vec<Matrix>,
    rt: Vec<Matrix>,
    lt: (BidiagonalMatrix, Vec<f64>),
    phantom: PhantomData<T>,
}

impl<K, T> KissLoveGP<K, T>
where
    K: Kernel<Vec<f64>>,
    T: Convolutable,
{
    pub fn from(kernel: K) -> Self {
        Self::new(Convolutional::new(kernel))
    }

    pub fn with_u(mut self, u: Grid) -> Self {
        self.u = u;

        self
    }

    fn reset_prepare(&mut self) -> Result<&mut Self, Box<dyn Error>> {
        self.ready_to_predict = false;
        self.a = vec![];
        self.rt = vec![];
        self.lt = (BidiagonalMatrix::default(), vec![]);

        Ok(self)
    }
}

impl<K, T> GaussianProcess<Convolutional<K>, T> for KissLoveGP<K, T>
where
    K: Kernel<Vec<f64>>,
    T: Convolutable,
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
            rt: vec![],
            lt: (BidiagonalMatrix::default(), vec![]),
            phantom: PhantomData::<T>,
        }
    }

    fn set_x(&mut self, x: Vec<T>) -> Result<&Self, Box<dyn Error>> {
        let n = x.len();
        if n == 0 {
            return Err(GaussianProcessError::Empty.into());
        }

        let parts_len = x[0].parts_len();
        let data_len = x[0].data_len();
        if parts_len == 0 || data_len == 0 {
            return Err(GaussianProcessError::Empty.into());
        }

        if self.u.axes().len() == 0 {
            let points = vec![(n / 2usize.pow(data_len as u32)).max(2); data_len];
            self.u = Grid::from(&x, &points)?;
            self.wx = self.u.interpolation_weight(&x)?;
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
        let wxt = &(0..p)
            .into_iter()
            .map(|pi| {
                let wxpi = &wx[pi];
                let wxpit = wxpi.t();

                wxpit
            })
            .collect::<Vec<_>>();

        let wxt_kuu_wx_sigma2_vec_mul = move |v: Vec<f64>| {
            (0..p)
                .into_iter()
                .map(|pi| {
                    let wxpi = &wx[pi];
                    let wxpit = &wxt[pi];

                    let v = v.clone().col_mat();
                    let wx_v = wxpi * &v;
                    let kzz_wx_v = kuu.vec_mul(wx_v.vec())?.col_mat();
                    let wxt_kzz_wx_v = wxpit * kzz_wx_v;
                    Ok(wxt_kzz_wx_v.vec())
                })
                .try_fold(vec![0.0; v.len()], |a, b: Result<_, Box<dyn Error>>| {
                    Ok((a.col_mat() + b?.col_mat()).vec())
                })
        };

        let wxt_kuu_wx_sigma2_inv_y =
            Matrix::posv_cgm(&wxt_kuu_wx_sigma2_vec_mul, y_ey.to_vec(), k)?.col_mat();

        // (wxt * kuu * wx + σ^2 I)^{-1} = q * t^{=1} * qt
        // q: n,k
        // t: k,k
        let (q, t) = Matrix::sytrd_k(n, k, &wxt_kuu_wx_sigma2_vec_mul, None)?;
        self.lt = t.pttrf()?;

        self.a = (0..p)
            .into_iter()
            .map(|pi| {
                let wxpi = &wx[pi];

                // a = kuu * wx * (wxt * kuu *wx + σ^2 I)^{-1} * y
                let a = kuu
                    .vec_mul((wxpi * &wxt_kuu_wx_sigma2_inv_y).vec())?
                    .col_mat();
                Ok(a)
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

        self.rt = (0..p)
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
                let r = Matrix::from(m, kuu_wx_r_cols.concat());

                /*
                // kzz - c = q * t * qt
                let (q, t) = Matrix::sytrd_k(
                    m,
                    100,
                    &|v| {
                        Ok((kzz.vec_mul(v)?.col_mat()
                            - &rt * l.pttrs(&d, v.to_vec().row_mat() * &rt)?.vec().col_mat())
                        .vec())
                    },
                    None,
                )?;

                // t = l * d * lt
                let (l, d) = t.pttrf()?;
                let lt = l.mat(true);
                let d_sqrt = Matrix::diag(&d).dipowf(0.5);
                */

                Ok(r.t())
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

        Ok(())
    }

    fn predict_multivariate(&self, xs: &[T]) -> Result<MultivariateNormalParams, Box<dyn Error>> {
        if !self.ready_to_predict {
            return Err(GaussianProcessError::NotPrepared.into());
        }

        let wxs = &self.u.interpolation_weight(xs)?;
        let p = self.a.len();

        if p != wxs.len() {
            return Err(GaussianProcessError::DimensionMismatch.into());
        }

        let (mean, wxst_rt) = (0..p)
            .into_iter()
            .map(|pi| {
                let wxspi = &wxs[pi];
                let wxspit = &wxspi.t();

                let a = &self.a[pi];
                let rt = &self.rt[pi];

                let mean = (self.ey + wxspit * a).vec();
                let wxst_rt = wxspit * rt;

                Ok((mean, wxst_rt))
            })
            .try_fold::<(Vec<f64>, Matrix), _, Result<(Vec<f64>, Matrix), Box<dyn Error>>>(
                (vec![], mat!()),
                |a, b: Result<(Vec<f64>, Matrix), Box<dyn Error>>| {
                    let b = b?;
                    Ok(((a.0.col_mat() + b.0.col_mat()).vec(), a.1 + b.1))
                },
            )?;

        let lt = &self.lt;

        let rprime_wx = lt.0.pttrs(&lt.1, wxst_rt.clone())?.t();
        let kxsxs = kernel_matrix(&self.kernel, &self.theta, xs, xs)?;

        let cov = kxsxs - wxst_rt * rprime_wx;

        MultivariateNormalParams::new(mean, cov.potrf()?)
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
        todo!()
    }

    fn sample(
        &self,
        theta: &Self::U,
        rng: &mut rand::prelude::StdRng,
    ) -> Result<Self::T, Box<dyn Error>> {
        todo!()
    }
}
