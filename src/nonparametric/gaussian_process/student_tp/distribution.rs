use super::{super::y_ey, GaussianProcess, StudentTP, StudentTPError};
use crate::{nonparametric::GaussianProcessParams, Distribution};
use opensrdk_kernel_method::Kernel;
use opensrdk_linear_algebra::*;
use rand::prelude::*;
use rand_distr::StudentT;
use special::Gamma;
use std::f64::consts::PI;
use std::fmt::Debug;

#[derive(Clone, Debug, PartialEq)]
pub struct StudentTPParams<T>
where
    T: Clone + Debug,
{
    pub x: Option<Vec<T>>,
    pub theta: Option<Vec<f64>>,
    pub nu: Option<f64>,
}

impl<G, K, T> Distribution for StudentTP<G, K, T>
where
    G: GaussianProcess<K, T>,
    K: Kernel<T>,
    T: Clone + Debug + PartialEq,
{
    type T = Vec<f64>;
    type U = StudentTPParams<T>;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, Box<dyn std::error::Error>> {
        let y = x;
        let n = self.gp.n();

        if y.len() != n {
            return Err(StudentTPError::DimensionMismatch.into());
        }
        let n = n as f64;
        let nu = self.nu;

        let y_ey = y_ey(y, self.gp.ey()).col_mat();

        let y_ey_t = y_ey.t();
        let (kxx_inv_y_ey, det) = self.gp.kxx_inv_vec(
            y_ey.vec(),
            &GaussianProcessParams {
                x: theta.x.clone(),
                theta: theta.theta.clone(),
            },
            true,
        )?;
        let (kxx_inv_y_ey, det) = (kxx_inv_y_ey.col_mat(), det.unwrap());

        Ok((Gamma::gamma((nu + n) / 2.0)
            / (Gamma::gamma(nu / 2.0) * nu.powf(n / 2.0) * PI.powf(n / 2.0) * det))
            * (1.0 + (y_ey_t * kxx_inv_y_ey)[0][0] / nu).powf(-(nu + n) / 2.0))
    }

    fn sample(
        &self,
        theta: &Self::U,
        rng: &mut rand::prelude::StdRng,
    ) -> Result<Self::T, Box<dyn std::error::Error>> {
        let n = self.gp.n();

        let student_t = StudentT::new(self.nu as f64)?;
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

        let mu = vec![self.gp.ey(); n].col_mat();
        let y = mu + wxt_lkuu_z;

        Ok(y.vec())
    }
}
