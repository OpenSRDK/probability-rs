use super::{GaussianProcess, StudentTP};
use crate::Distribution;
use opensrdk_kernel_method::Kernel;
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
        todo!()
    }

    fn sample(
        &self,
        theta: &Self::U,
        rng: &mut rand::prelude::StdRng,
    ) -> Result<Self::T, Box<dyn std::error::Error>> {
        todo!()
    }
}
