pub mod distribution;
pub mod grid;
pub mod internal;
pub mod regressor;

use super::{GaussianProcess, GaussianProcessError, GaussianProcessParams};
use crate::{opensrdk_linear_algebra::*, RandomVariable};
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
#[derive(Clone, Debug)]
pub struct KissLoveGP<K, T>
where
    K: Kernel<Vec<f64>>,
    T: RandomVariable + Convolutable,
{
    kernel: Convolutional<K>,
    phantom: PhantomData<T>,
}

impl<K, T> KissLoveGP<K, T>
where
    K: Kernel<Vec<f64>>,
    T: RandomVariable + Convolutable,
{
    pub fn from(kernel: K) -> Self {
        Self::new(Convolutional::new(kernel))
    }
}

impl<K, T> GaussianProcess<Convolutional<K>, T> for KissLoveGP<K, T>
where
    K: Kernel<Vec<f64>>,
    T: RandomVariable + Convolutable,
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
        Self {
            kernel,
            phantom: PhantomData::<T>,
        }
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
        let n = params.x.len();
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
