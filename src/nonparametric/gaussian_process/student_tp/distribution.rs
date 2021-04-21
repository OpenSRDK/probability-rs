use super::{GaussianProcess, StudentTP, StudentTPError};
use crate::DistributionError;
use crate::{nonparametric::GaussianProcessParams, Distribution, RandomVariable};
use opensrdk_kernel_method::Kernel;
use opensrdk_linear_algebra::*;
use rand::prelude::*;
use rand_distr::StudentT;
use special::Gamma;
use std::fmt::Debug;
use std::{error::Error, f64::consts::PI};

#[derive(Clone, Debug, PartialEq)]
pub struct StudentTPParams<T>
where
  T: RandomVariable,
{
  x: Vec<T>,
  theta: Vec<f64>,
  nu: f64,
}

impl<T> StudentTPParams<T>
where
  T: RandomVariable,
{
  pub fn new(x: Vec<T>, theta: Vec<f64>, nu: f64) -> Result<Self, Box<dyn Error>> {
    if nu <= 0.0 {
      return Err(StudentTPError::NuMustBePositive.into());
    }

    Ok(Self { x, theta, nu })
  }

  pub fn eject(self) -> (Vec<T>, Vec<f64>, f64) {
    (self.x, self.theta, self.nu)
  }
}

impl<G, K, T> Distribution for StudentTP<G, K, T>
where
  G: GaussianProcess<K, T>,
  K: Kernel<T>,
  T: RandomVariable,
{
  type T = Vec<f64>;
  type U = StudentTPParams<T>;

  fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
    let y = x;
    let n = y.len();

    let y_ey = y.clone().col_mat();
    let y_ey_t = y_ey.t();

    let n = n as f64;
    let nu = theta.nu;

    let (kxx_inv_y_ey, det) = self.gp.kxx_inv_vec(
      y_ey.vec(),
      &GaussianProcessParams {
        x: theta.x.clone(),
        theta: theta.theta.clone(),
      },
      true,
    )?;
    let (kxx_inv_y_ey, det) = (kxx_inv_y_ey.col_mat(), det.unwrap());

    Ok(
      (Gamma::gamma((nu + n) / 2.0)
        / (Gamma::gamma(nu / 2.0) * nu.powf(n / 2.0) * PI.powf(n / 2.0) * det))
        * (1.0 + (y_ey_t * kxx_inv_y_ey)[0][0] / nu).powf(-(nu + n) / 2.0),
    )
  }

  fn sample(
    &self,
    theta: &Self::U,
    rng: &mut rand::prelude::StdRng,
  ) -> Result<Self::T, DistributionError> {
    let n = theta.x.len();

    let student_t = match StudentT::new(theta.nu) {
      Ok(v) => Ok(v),
      Err(e) => Err(DistributionError::Others(e.into())),
    }?;
    let z = (0..n)
      .into_iter()
      .map(|_| rng.sample(student_t))
      .collect::<Vec<_>>();

    let wxt_lkuu_z = self
      .gp
      .lkxx_vec(
        z,
        &GaussianProcessParams {
          x: theta.x.clone(),
          theta: theta.theta.clone(),
        },
      )?
      .col_mat();

    let y = wxt_lkuu_z;

    Ok(y.vec())
  }
}
