pub mod dirichlet_process;

pub use dirichlet_process::*;

use crate::{Distribution, RandomVariable};
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct BaselineMeasure<D, T>
where
    D: Distribution<T = T, U = ()>,
    T: RandomVariable,
{
    pub distr: D,
}

impl<D, T> BaselineMeasure<D, T>
where
    D: Distribution<T = T, U = ()>,
    T: RandomVariable,
{
    pub fn new(distr: D) -> Self {
        Self { distr }
    }
}

pub type DiscreteMeasurableSpace = HashSet<usize>;

pub trait DiscreteMeasure {
    fn measure(&self, a: DiscreteMeasurableSpace) -> f64;
}
