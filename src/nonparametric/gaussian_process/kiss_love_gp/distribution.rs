use crate::{
    nonparametric::{y_ey, GaussianProcessError, GaussianProcessParams},
    opensrdk_linear_algebra::*,
    Distribution,
};
use opensrdk_kernel_method::{Convolutable, Kernel};
use rand::Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::{error::Error, f64::consts::PI};

use super::KissLoveGP;

impl<K, T> Distribution for KissLoveGP<K, T>
where
    K: Kernel<Vec<f64>>,
    T: Convolutable + PartialEq,
{
    type T = Vec<f64>;
    type U = GaussianProcessParams<T>;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, Box<dyn Error>> {
        let y = x;
        let (wx, kuu) = self.handle_temporal_params(theta)?;
        let n = wx[0].cols;
        let k = n.min(100);

        if y.len() != n {
            return Err(GaussianProcessError::DimensionMismatch.into());
        }

        let det = Self::det_kxx(&kuu, &wx)?;
        let y_ey = y_ey(y, self.ey).col_mat();

        let wxt_kuu_wx_vec_mul = move |v: Vec<f64>| Self::wxt_kuu_wx_vec_mul(&v, &wx, &kuu);

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
        let (wx, kuu) = self.handle_temporal_params(theta)?;
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
