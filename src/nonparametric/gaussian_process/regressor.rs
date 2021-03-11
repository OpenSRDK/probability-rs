use std::{error::Error, fmt::Debug};

use opensrdk_kernel_method::Kernel;

use crate::{MultivariateNormalParams, NormalParams, RandomVariable};

use super::{GaussianProcess, GaussianProcessParams};

#[derive(thiserror::Error, Debug)]
pub enum GaussianProcessRegressorError {
    #[error("Data is empty.")]
    Empty,
    #[error("Dimension mismatch.")]
    DimensionMismatch,
    #[error("NaN contaminated.")]
    NaNContamination,
}

fn ref_to_slice<T>(v: &T) -> &[T] {
    unsafe { ::std::slice::from_raw_parts(v as *const T, 1) }
}

pub trait GaussianProcessRegressor<G, K, T>: Sized
where
    G: GaussianProcess<K, T>,
    K: Kernel<T>,
    T: RandomVariable,
{
    fn new(gp: G, y: &[f64], params: GaussianProcessParams<T>) -> Result<Self, Box<dyn Error>>;

    fn n(&self) -> usize;
    fn ey(&self) -> f64;

    fn predict(&self, xs: &T) -> Result<NormalParams, Box<dyn Error>> {
        let mul_n = self.predict_multivariate(ref_to_slice(xs))?;

        NormalParams::new(mul_n.mu()[0], mul_n.lsigma()[0][0])
    }

    fn predict_multivariate(&self, xs: &[T]) -> Result<MultivariateNormalParams, Box<dyn Error>>;
}
