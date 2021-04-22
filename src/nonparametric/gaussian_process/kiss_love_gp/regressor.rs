use super::{
  super::{ey::ey, ey::y_ey},
  grid::Grid,
  KissLoveGP,
};
use crate::DistributionError;
use crate::{
  nonparametric::regressor::{GaussianProcessRegressor, GaussianProcessRegressorError},
  RandomVariable,
};
use crate::{nonparametric::GaussianProcessParams, MultivariateNormalParams};
use opensrdk_kernel_method::{Convolutable, Convolutional, Kernel};
use opensrdk_linear_algebra::*;
use std::marker::PhantomData;

pub struct KissLoveGPregressor<K, T>
where
  K: Kernel<Vec<f64>>,
  T: RandomVariable + Convolutable,
{
  n: usize,
  ey: f64,
  u: Grid,
  a: Vec<Matrix>,
  s: Vec<Matrix>,
  phantom: PhantomData<(K, T)>,
}

impl<K, T> GaussianProcessRegressor<KissLoveGP<K, T>, Convolutional<K>, T>
  for KissLoveGPregressor<K, T>
where
  K: Kernel<Vec<f64>>,
  T: RandomVariable + Convolutable,
{
  fn new(
    gp: KissLoveGP<K, T>,
    y: &[f64],
    params: GaussianProcessParams<T>,
  ) -> Result<Self, DistributionError> {
    let (x, theta) = params.eject();

    let n = y.len();
    if n == 0 {
      return Err(DistributionError::InvalidParameters(
        GaussianProcessRegressorError::Empty.into(),
      ));
    }

    if n != x.len() {
      return Err(DistributionError::InvalidParameters(
        GaussianProcessRegressorError::DimensionMismatch.into(),
      ));
    }

    let (wx, u) = KissLoveGP::<K, T>::wx_u(&x)?;
    let wx = &wx;
    let kuu = &u.kuu(&gp.kernel, &theta)?;

    let m = kuu.rows();
    const K: usize = 100;
    let k = n.min(K);
    let p = wx.len();

    let ey = ey(y);
    let y_ey = &y_ey(y, ey);

    let wxt_kuu_wx_vec_mul =
      move |v: Vec<f64>| match KissLoveGP::<K, T>::wxt_kuu_wx_vec_mul(&v, wx, kuu) {
        Ok(v) => Ok(v),
        Err(e) => Err(e.into()),
      };

    let wxt_kuu_wx_inv_y = Matrix::posv_cgm(&wxt_kuu_wx_vec_mul, y_ey.to_vec(), K)?.col_mat();

    let a = (0..p)
      .into_iter()
      .map(|pi| {
        let wxpi = &wx[pi];

        // a = kuu * wx * (wxt * kuu *wx)^{-1} * y
        let a = kuu.vec_mul((wxpi * &wxt_kuu_wx_inv_y).vec())?.col_mat();
        Ok(a)
      })
      .collect::<Result<Vec<_>, DistributionError>>()?;

    // (wxt * kuu * wx)^{-1} = q * t^{-1} * qt
    // q: n×k
    // t: k×k
    let (q, t) = Matrix::sytrd_k(n, k, &wxt_kuu_wx_vec_mul, None)?;

    // t = l * d * lt
    let (l, d) = t.pttrf()?;

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
        let r = Matrix::from(m, kuu_wx_r_cols.concat());

        // kuu - rt * (l * d * lt)^{-1} * r = q2 * t2 * q2t
        let (q2, t2) = Matrix::sytrd_k(
          m,
          100,
          &|v| {
            Ok(
              (kuu.vec_mul(v.clone())?.col_mat()
                - r.t() * l.pttrs(&d, &r * v.col_mat())?.vec().col_mat())
              .vec(),
            )
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
      .collect::<Result<Vec<_>, DistributionError>>()?;

    Ok(Self {
      n,
      ey,
      u,
      a,
      s,
      phantom: PhantomData,
    })
  }

  fn n(&self) -> usize {
    self.n
  }

  fn ey(&self) -> f64 {
    self.ey
  }

  fn predict_multivariate(&self, xs: &[T]) -> Result<MultivariateNormalParams, DistributionError> {
    let len = xs.len();
    if len == 0 {
      return Err(DistributionError::InvalidParameters(
        GaussianProcessRegressorError::Empty.into(),
      ));
    }

    let wxs = &self.u.interpolation_weight(xs)?;
    let p = self.a.len();

    if p != wxs.len() {
      return Err(DistributionError::InvalidParameters(
        GaussianProcessRegressorError::DimensionMismatch.into(),
      ));
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
      .try_fold::<(Vec<f64>, Matrix), _, Result<(Vec<f64>, Matrix), DistributionError>>(
        (vec![self.ey; len], Matrix::new(len, len)),
        |a, b: Result<(Vec<f64>, Matrix), DistributionError>| {
          let b = b?;
          Ok(((a.0.col_mat() + b.0.col_mat()).vec(), a.1 + b.1))
        },
      )?;

    MultivariateNormalParams::new(mu, l_sigma)
  }
}
