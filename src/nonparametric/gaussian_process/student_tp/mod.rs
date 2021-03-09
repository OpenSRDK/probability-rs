pub mod distribution;

use super::GaussianProcess;
use crate::MultivariateStudentTParams;
use opensrdk_kernel_method::Kernel;
use std::{error::Error, fmt::Debug, marker::PhantomData};

#[derive(thiserror::Error, Debug)]
pub enum StudentTPError {
    #[error("'ν' must be positive")]
    NuMustBePositive,
}

pub struct StudentTP<G, K, T>
where
    G: GaussianProcess<K, T>,
    K: Kernel<T>,
    T: Clone + Debug + PartialEq,
{
    gp: G,
    nu: f64,
    phantom: PhantomData<(K, T)>,
}

impl<G, K, T> StudentTP<G, K, T>
where
    G: GaussianProcess<K, T>,
    K: Kernel<T>,
    T: Clone + Debug + PartialEq,
{
    pub fn set_nu(&mut self, nu: f64) -> Result<&Self, Box<dyn Error>> {
        if nu <= 0.0 {
            return Err(StudentTPError::NuMustBePositive.into());
        }
        self.nu = nu;

        Ok(self)
    }

    pub fn nu(&self) -> f64 {
        self.nu
    }

    pub fn predict_multivariate(
        &self,
        xs: &[T],
    ) -> Result<MultivariateStudentTParams, Box<dyn Error>> {
        let params = self.gp.predict_multivariate(xs)?;
        let nu = self.nu;
        let beta = 0.0;
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
