use super::{distribution::StudentTPParams, StudentTP};
use crate::{
    nonparametric::{GaussianProcess, GaussianProcessParams, GaussianProcessRegressor},
    MultivariateStudentTParams, RandomVariable,
};
use opensrdk_kernel_method::Kernel;
use opensrdk_linear_algebra::*;
use std::{error::Error, marker::PhantomData};

pub struct StudentTPRegressor<G, R, K, T>
where
    G: GaussianProcess<K, T>,
    R: GaussianProcessRegressor<G, K, T>,
    K: Kernel<T>,
    T: RandomVariable,
{
    gpr: R,
    nu: f64,
    beta: f64,
    phantom: PhantomData<(G, K, T)>,
}

impl<G, R, K, T> StudentTPRegressor<G, R, K, T>
where
    G: GaussianProcess<K, T>,
    R: GaussianProcessRegressor<G, K, T>,
    K: Kernel<T>,
    T: RandomVariable,
{
    pub fn new(
        tp: StudentTP<G, K, T>,
        y: &[f64],
        params: StudentTPParams<T>,
    ) -> Result<Self, Box<dyn Error>> {
        let (x, theta, nu) = params.eject();
        let params = GaussianProcessParams { x, theta };
        let kxx_inv_y = tp.gp.kxx_inv_vec(y.to_vec(), &params, false)?.0.col_mat();
        let yt = y.to_vec().row_mat();

        let gpr = R::new(tp.gp, y, params)?;

        let beta = (yt * kxx_inv_y)[0][0];

        Ok(Self {
            gpr,
            nu,
            beta,
            phantom: PhantomData,
        })
    }

    pub fn gpr(&self) -> &R {
        &self.gpr
    }

    pub fn nu(&self) -> f64 {
        self.nu
    }

    pub fn beta(&self) -> f64 {
        self.beta
    }

    pub fn predict_multivariate(
        &self,
        xs: &[T],
    ) -> Result<MultivariateStudentTParams, Box<dyn Error>> {
        let params = self.gpr.predict_multivariate(xs)?;
        let nu = self.nu;
        let beta = self.beta;

        let m = xs.len() as f64;
        let n = self.gpr.n() as f64;

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
