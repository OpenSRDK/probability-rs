use super::KissLoveGP;
use crate::{
    nonparametric::{y_ey, GaussianProcess, GaussianProcessError, GaussianProcessParams},
    opensrdk_linear_algebra::*,
    Distribution,
};
use opensrdk_kernel_method::{Convolutable, Kernel};
use rand::Rng;
use rand_distr::StandardNormal;
use std::{error::Error, f64::consts::PI};

impl<K, T> Distribution for KissLoveGP<K, T>
where
    K: Kernel<Vec<f64>>,
    T: Convolutable + PartialEq,
{
    type T = Vec<f64>;
    type U = GaussianProcessParams<T>;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, Box<dyn Error>> {
        let y = x;
        let n = self.n();

        if y.len() != n {
            return Err(GaussianProcessError::DimensionMismatch.into());
        }

        let y_ey = y_ey(y, self.ey).col_mat();

        let y_ey_t = y_ey.t();
        let (kxx_inv_y_ey, det) = self.kxx_inv_vec(y_ey.vec(), theta, true)?;
        let (kxx_inv_y_ey, det) = (kxx_inv_y_ey.col_mat(), det.unwrap());

        Ok(1.0 / ((2.0 * PI).powf(n as f64 / 2.0) * det)
            * (-1.0 / 2.0 * (y_ey_t * kxx_inv_y_ey)[0][0]).exp())
    }

    fn sample(
        &self,
        theta: &Self::U,
        rng: &mut rand::prelude::StdRng,
    ) -> Result<Self::T, Box<dyn Error>> {
        let n = self.n();
        let z = (0..n)
            .into_iter()
            .map(|_| rng.sample(StandardNormal))
            .collect::<Vec<_>>();

        let wxt_lkuu_z = self.lkxx_vec(z, theta)?.col_mat();

        let mu = vec![self.ey; n].col_mat();
        let y = mu + wxt_lkuu_z;

        Ok(y.vec())
    }
}
