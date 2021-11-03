use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use std::{
    collections::HashMap,
    fmt::Debug,
    ops::{BitAnd, Mul},
};

#[derive(Clone, Debug)]
pub struct SwitchedDistribution<D, T, U>
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    distribution: D,
}

#[derive(thiserror::Error, Debug)]
pub enum SwitchedError {
    #[error("Key not found")]
    KeyNotFound,
    #[error("None is invalid for sample.")]
    NoneIsInvalidForSample,
    #[error("Unknown error")]
    Unknown,
}

impl<D, T, U> SwitchedDistribution<D, T, U>
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    pub fn new(distribution: D) -> Self {
        Self { distribution }
    }
}

impl<D, T, U> Distribution for SwitchedDistribution<D, T, U>
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    type T = T;
    type U = SwitchedParams<U>;

    fn fk(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let s = theta;

        match s {
            SwitchedParams::Key(k, map) => match map.get(k) {
                Some(theta) => self.distribution.fk(x, theta),
                None => Err(DistributionError::InvalidParameters(
                    SwitchedError::KeyNotFound.into(),
                )),
            },
            SwitchedParams::Direct(theta) => self.distribution.fk(x, theta),
            SwitchedParams::None => Ok(1.0),
        }
    }

    fn sample(&self, theta: &Self::U, rng: &mut dyn RngCore) -> Result<Self::T, DistributionError> {
        let s = theta;

        match s {
            SwitchedParams::Key(k, map) => match map.get(k) {
                Some(theta) => self.distribution.sample(theta, rng),
                None => Err(DistributionError::InvalidParameters(
                    SwitchedError::KeyNotFound.into(),
                )),
            },
            SwitchedParams::Direct(theta) => self.distribution.sample(theta, rng),
            SwitchedParams::None => Err(DistributionError::InvalidParameters(
                SwitchedError::NoneIsInvalidForSample.into(),
            )),
        }
    }
}

impl<D, T, U> SwitchedDistribution<D, T, U>
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    pub fn distribution(&self) -> &D {
        &self.distribution
    }
}

#[derive(Clone, Debug)]
pub enum SwitchedParams<U>
where
    U: RandomVariable,
{
    Key(u32, HashMap<u32, U>),
    Direct(U),
    None,
}

pub trait SwitchableDistribution: Distribution + Sized {
    fn switch<'a>(self) -> SwitchedDistribution<Self, Self::T, Self::U>;
}

impl<D, T, U> SwitchableDistribution for D
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    fn switch(self) -> SwitchedDistribution<Self, Self::T, U> {
        SwitchedDistribution::<Self, Self::T, U>::new(self)
    }
}

impl<D, T, U, Rhs, TRhs> Mul<Rhs> for SwitchedDistribution<D, T, U>
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

impl<D, T, U, Rhs, URhs> BitAnd<Rhs> for SwitchedDistribution<D, T, U>
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

#[cfg(test)]
mod tests {
    use crate::*;
    use std::collections::HashMap;

    #[test]
    fn it_works() {
        let distr = Normal.switch();
        let mut theta = HashMap::new();
        theta.insert(1u32, NormalParams::new(1.0, 2.0).unwrap());
        theta.insert(2u32, NormalParams::new(2.0, 2.0).unwrap());
        theta.insert(3u32, NormalParams::new(3.0, 2.0).unwrap());
        theta.insert(4u32, NormalParams::new(4.0, 2.0).unwrap());
        let switched_fk = distr.fk(&0f64, &SwitchedParams::Key(1u32, theta)).unwrap();
        let fk = Normal
            .fk(&0f64, &NormalParams::new(1.0, 2.0).unwrap())
            .unwrap();

        assert_eq!(switched_fk, fk);
    }
}
