use super::{grid::Grid, GaussianProcessError, GaussianProcessParams, KissLoveGP};
use crate::{opensrdk_linear_algebra::*, RandomVariable};
use opensrdk_kernel_method::{Convolutable, Kernel};
use rayon::prelude::*;
use std::{cmp::Ordering, error::Error};

impl<K, T> KissLoveGP<K, T>
where
  K: Kernel<Vec<f64>>,
  T: RandomVariable + Convolutable,
{
  pub(crate) fn wx_u(x: &Vec<T>) -> Result<(Vec<SparseMatrix>, Grid), Box<dyn Error>> {
    let n = x.len();
    if n == 0 {
      return Err(GaussianProcessError::Empty.into());
    }

    let parts_len = x[0].parts_len();
    let data_len = x[0].data_len();
    if parts_len == 0 || data_len == 0 {
      return Err(GaussianProcessError::Empty.into());
    }

    let points = vec![(n / 2usize.pow(data_len as u32)).max(2); data_len];
    let u = Grid::from(&x, &points)?;
    let wx = u.interpolation_weight(&x)?;

    return Ok((wx, u));
  }

  pub(crate) fn wxt_kuu_wx_vec_mul(
    v: &Vec<f64>,
    wx: &Vec<SparseMatrix>,
    kuu: &KroneckerMatrices,
  ) -> Result<Vec<f64>, Box<dyn Error>> {
    wx.iter()
      .map(|wxpi| {
        let v = v.clone().col_mat();
        let wx_v = wxpi * &v;
        let kuu_wx_v = kuu.vec_mul(wx_v.vec())?.col_mat();
        let wxt_kuu_wx_v = wxpi.t() * kuu_wx_v;
        Ok(wxt_kuu_wx_v.vec())
      })
      .try_fold(vec![0.0; v.len()], |a, b: Result<_, Box<dyn Error>>| {
        Ok((a.col_mat() + b?.col_mat()).vec())
      })
  }

  pub(crate) fn handle_temporal_params(
    &self,
    params: &GaussianProcessParams<T>,
  ) -> Result<(Vec<SparseMatrix>, KroneckerMatrices), Box<dyn Error>> {
    let (wx, u) = Self::wx_u(&params.x)?;
    let kuu = u.kuu(&self.kernel, &params.theta)?;

    return Ok((wx, kuu));
  }

  /// See Andrew Gordon Wilson
  pub(crate) fn det_kxx(
    kuu: &KroneckerMatrices,
    wx: &Vec<SparseMatrix>,
  ) -> Result<f64, Box<dyn Error>> {
    let m = wx[0].rows;
    let n = wx[0].cols;

    let kuu_toeplitz = kuu
      .matrices()
      .iter()
      .map(|kp| Ok(ToeplitzMatrix::new(kp[0].to_vec(), kp[0][1..].to_vec())?))
      .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    let lambda = kuu_toeplitz
      .par_iter()
      .map(|kp| {
        kp.embedded_circulant().cigv().1[..kp.dim()]
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
      return Err(GaussianProcessError::NaNContamination.into());
    }

    let lambda = &lambda[m - n..];

    let det = lambda
      .par_iter()
      .map(|lmd| ((n as f64 / m as f64) * lmd.re))
      .product::<f64>();

    Ok(det)
  }

  pub(crate) fn lkuu(kuu: KroneckerMatrices) -> Result<KroneckerMatrices, Box<dyn Error>> {
    let matrices = kuu.eject();
    Ok(KroneckerMatrices::new(
      matrices
        .into_iter()
        .map(|kpi| kpi.potrf())
        .collect::<Result<Vec<_>, _>>()?,
    ))
  }
}
