use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use std::{
    collections::HashMap,
    fmt::Debug,
    ops::{BitAnd, Mul},
};

#[derive(Clone, Debug)]
pub struct SwitchedDistribution<'a, D, T, U>
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    distribution: D,
    map: &'a HashMap<u32, U>,
}

#[derive(thiserror::Error, Debug)]
pub enum SwitchedError {
    #[error("Key not found")]
    KeyNotFound,
    #[error("Unknown error")]
    Unknown,
}

impl<'a, D, T, U> SwitchedDistribution<'a, D, T, U>
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    pub fn new(distribution: D, map: &'a HashMap<u32, U>) -> Self {
        Self { distribution, map }
    }
}

impl<'a, D, T, U> Distribution for SwitchedDistribution<'a, D, T, U>
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    type T = T;
    type U = SwitchedParams<U>;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let s = theta;

        match s {
            SwitchedParams::Key(k) => match self.map.get(k) {
                Some(theta) => self.distribution.p(x, theta),
                None => Err(DistributionError::InvalidParameters(
                    SwitchedError::KeyNotFound.into(),
                )),
            },
            SwitchedParams::Direct(theta) => self.distribution.p(x, theta),
        }
    }

    fn sample(&self, theta: &Self::U, rng: &mut dyn RngCore) -> Result<Self::T, DistributionError> {
        let s = theta;

        match s {
            SwitchedParams::Key(k) => match self.map.get(k) {
                Some(theta) => self.distribution.sample(theta, rng),
                None => Err(DistributionError::InvalidParameters(
                    SwitchedError::KeyNotFound.into(),
                )),
            },
            SwitchedParams::Direct(theta) => self.distribution.sample(theta, rng),
        }
    }
}

#[derive(Clone, Debug)]
pub enum SwitchedParams<U>
where
    U: RandomVariable,
{
    Key(u32),
    Direct(U),
}

pub trait SwitchableDistribution: Distribution + Sized {
    fn switch<'a>(
        self,
        map: &'a HashMap<u32, Self::U>,
    ) -> SwitchedDistribution<'a, Self, Self::T, Self::U>;
}

impl<D, T, U> SwitchableDistribution for D
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    fn switch<'a>(self, map: &'a HashMap<u32, U>) -> SwitchedDistribution<'a, Self, Self::T, U> {
        SwitchedDistribution::<Self, Self::T, U>::new(self, map)
    }
}

impl<'a, D, T, U, Rhs, TRhs> Mul<Rhs> for SwitchedDistribution<'a, D, T, U>
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
    Rhs: Distribution<T = TRhs, U = SwitchedParams<U>>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, T, TRhs, SwitchedParams<U>>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<'a, D, T, U, Rhs, URhs> BitAnd<Rhs> for SwitchedDistribution<'a, D, T, U>
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
    Rhs: Distribution<T = SwitchedParams<U>, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, T, SwitchedParams<U>, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}
