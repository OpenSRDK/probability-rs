use rand::prelude::*;
use std::{error::Error, fmt::Debug};

use crate::LogDistribution;

#[derive(thiserror::Error, Debug)]
pub enum DistributionError {
    #[error("params are not set")]
    ParamsAreNotSet,
}

/// # Distribution
/// ![tex](https://latex.codecogs.com/svg.latex?P%28x%7c\mathbf{\theta}%29)
pub trait Distribution<'a, T>
where
    T: Clone + Debug,
{
    fn p(&self, x: &T) -> Result<f64, Box<dyn Error>>;
    fn sample(&self, rng: &mut StdRng) -> Result<T, Box<dyn Error>>;
    fn ln(&'a mut self, x: &'a mut DistributionParamVal<T>) -> LogDistribution<'a>;
}

pub trait DistributionParam<T>: Debug
where
    T: Debug,
{
    fn value(&self) -> &T;
    fn mut_for_optimization(&mut self) -> Option<&mut T>;
    fn ref_for_optimization(&self) -> Option<&T>;
}

#[derive(Clone, Debug)]
pub struct DistributionParamVal<T>
where
    T: Clone + Debug,
{
    value: T,
    optimization: bool,
}

impl<T> DistributionParamVal<T>
where
    T: Clone + Debug,
{
    pub fn new(value: T) -> Self {
        Self {
            value,
            optimization: false,
        }
    }

    pub fn with_optimization(mut self) -> Self {
        self.optimization = true;

        self
    }
}

impl<T> DistributionParam<T> for DistributionParamVal<T>
where
    T: Clone + Debug,
{
    fn value(&self) -> &T {
        &self.value
    }

    fn mut_for_optimization(&mut self) -> Option<&mut T> {
        if !self.optimization {
            return None;
        }

        Some(&mut self.value)
    }

    fn ref_for_optimization(&self) -> Option<&T> {
        None
    }
}

#[derive(Clone, Debug)]
pub struct DistributionParamRef<'a, T>
where
    T: Clone + Debug,
{
    value: &'a DistributionParamVal<T>,
}

impl<'a, T> DistributionParamRef<'a, T>
where
    T: Clone + Debug,
{
    pub fn new(value: &'a DistributionParamVal<T>) -> Self {
        Self { value }
    }
}

impl<'a, T> DistributionParam<T> for DistributionParamRef<'a, T>
where
    T: Clone + Debug,
{
    fn value(&self) -> &T {
        &self.value.value()
    }

    fn mut_for_optimization(&mut self) -> Option<&mut T> {
        None
    }

    fn ref_for_optimization(&self) -> Option<&T> {
        if !self.value.optimization {
            return None;
        }

        Some(self.value())
    }
}
