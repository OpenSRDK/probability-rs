pub mod distribution;

use super::{GaussianProcess, GaussianProcessParams};
use crate::MultivariateStudentTParams;
use opensrdk_kernel_method::Kernel;
use opensrdk_linear_algebra::Vector;
use std::{error::Error, fmt::Debug, marker::PhantomData};

#[derive(thiserror::Error, Debug)]
pub enum StudentTPError {
    #[error("Dimension mismatch.")]
    DimensionMismatch,
    #[error("'ν' must be positive")]
    NuMustBePositive,
    #[error("Not prepared.")]
    NotPrepared,
}

pub struct StudentTP<G, K, T>
where
    G: GaussianProcess<K, T>,
    K: Kernel<T>,
    T: Clone + Debug + PartialEq,
{
    gp: G,
    nu: f64,
    ready_to_predict: bool,
    beta: f64,
    phantom: PhantomData<(K, T)>,
}

impl<G, K, T> StudentTP<G, K, T>
where
    G: GaussianProcess<K, T>,
    K: Kernel<T>,
    T: Clone + Debug + PartialEq,
{
    pub fn new(gp: G) -> Self {
        Self {
            gp,
            nu: 1.0,
            ready_to_predict: false,
            beta: 0.0,
            phantom: PhantomData,
        }
    }

    pub fn set_x(&mut self, x: Vec<T>) -> Result<&mut Self, Box<dyn Error>> {
        self.gp.set_x(x)?;
        self.reset_prepare();

        Ok(self)
    }

    pub fn set_theta(&mut self, theta: Vec<f64>) -> Result<&mut Self, Box<dyn Error>> {
        self.gp.set_theta(theta)?;
        self.reset_prepare();

        Ok(self)
    }

    pub fn set_nu(&mut self, nu: f64) -> Result<&Self, Box<dyn Error>> {
        if nu <= 0.0 {
            return Err(StudentTPError::NuMustBePositive.into());
        }
        self.nu = nu;
        self.reset_prepare();

        Ok(self)
    }

    pub fn gp(&self) -> &G {
        &self.gp
    }

    pub fn nu(&self) -> f64 {
        self.nu
    }

    fn reset_prepare(&mut self) {
        self.ready_to_predict = false;
        self.beta = 0.0;
    }

    pub fn prepare_predict(&mut self, y: &[f64]) -> Result<(), Box<dyn Error>> {
        self.gp.prepare_predict(y)?;

        let kxx_inv_y = self
            .gp
            .kxx_inv_vec(
                y.to_vec(),
                &GaussianProcessParams {
                    x: None,
                    theta: None,
                },
                false,
            )?
            .0
            .col_mat();
        let yt = y.to_vec().row_mat();
        self.beta = (yt * kxx_inv_y)[0][0];

        self.ready_to_predict = true;

        Ok(())
    }

    pub fn predict_multivariate(
        &self,
        xs: &[T],
    ) -> Result<MultivariateStudentTParams, Box<dyn Error>> {
        if !self.ready_to_predict {
            return Err(StudentTPError::NotPrepared.into());
        }
        let params = self.gp.predict_multivariate(xs)?;
        let nu = self.nu;
        let beta = self.beta;

        let m = xs.len() as f64;
        let n = self.gp.n() as f64;

        let coefficient = (nu - 1.0 - beta - 1.0) / (nu - 1.0 - n - 1.0);

        let (mu, l_sigma) = params.eject();
        let new_mu = mu;
        let new_l_sigma = coefficient.sqrt() * l_sigma;
        let new_nu = nu + m;

        Ok(MultivariateStudentTParams::new(
            new_mu,
            new_l_sigma,
            new_nu,
        )?)
    }
}
