pub mod distribution;
pub mod regressor;

use super::GaussianProcess;
use crate::RandomVariable;
use opensrdk_kernel_method::Kernel;
use std::{fmt::Debug, marker::PhantomData};

#[derive(thiserror::Error, Debug)]
pub enum StudentTPError {
    #[error("Dimension mismatch.")]
    DimensionMismatch,
    #[error("'ν' must be positive")]
    NuMustBePositive,
}

pub struct StudentTP<G, K, T>
where
    G: GaussianProcess<K, T>,
    K: Kernel<T>,
    T: RandomVariable,
{
    gp: G,
    phantom: PhantomData<(K, T)>,
}

impl<G, K, T> StudentTP<G, K, T>
where
    G: GaussianProcess<K, T>,
    K: Kernel<T>,
    T: RandomVariable,
{
    pub fn new(gp: G) -> Self {
        Self {
            gp,
            phantom: PhantomData,
        }
    }
}
