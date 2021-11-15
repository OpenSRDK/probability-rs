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
    D: Distribution<Value = T, Condition = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    distribution: &'a D,
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
    D: Distribution<Value = T, Condition = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    pub fn new(distribution: &'a D, map: &'a HashMap<u32, U>) -> Self {
        Self { distribution, map }
    }
}

impl<'a, D, T, U> Distribution for SwitchedDistribution<'a, D, T, U>
where
    D: Distribution<Value = T, Condition = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    type Value = T;
    type Condition = SwitchedParams<U>;

    fn fk(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let s = theta;

        match s {
            SwitchedParams::Key(k) => match self.map.get(k) {
                Some(theta) => self.distribution.fk(x, theta),
                None => Err(DistributionError::InvalidParameters(
                    SwitchedError::KeyNotFound.into(),
                )),
            },
            SwitchedParams::Direct(theta) => self.distribution.fk(x, theta),
        }
    }

    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
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

impl<'a, D, T, U> SwitchedDistribution<'a, D, T, U>
where
    D: Distribution<Value = T, Condition = U>,
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
    Key(u32),
    Direct(U),
}

pub trait SwitchableDistribution<U>: Distribution + Sized
where
    U: RandomVariable,
{
    fn switch<'a>(
        &'a self,
        map: &'a HashMap<u32, U>,
    ) -> SwitchedDistribution<'a, Self, Self::Value, Self::Condition>;
}

impl<D, T, U> SwitchableDistribution<U> for D
where
    D: Distribution<Value = T, Condition = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    fn switch<'a>(
        &'a self,
        map: &'a HashMap<u32, U>,
    ) -> SwitchedDistribution<'a, Self, Self::Value, U> {
        SwitchedDistribution::<Self, Self::Value, U>::new(self, map)
    }
}

impl<'a, D, T, U, Rhs, TRhs> Mul<Rhs> for SwitchedDistribution<'a, D, T, U>
where
    D: Distribution<Value = T, Condition = U>,
    T: RandomVariable,
    U: RandomVariable,
    Rhs: Distribution<Value = TRhs, Condition = SwitchedParams<U>>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, T, TRhs, SwitchedParams<U>>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<'a, D, T, U, Rhs, URhs> BitAnd<Rhs> for SwitchedDistribution<'a, D, T, U>
where
    D: Distribution<Value = T, Condition = U>,
    T: RandomVariable,
    U: RandomVariable,
    Rhs: Distribution<Value = SwitchedParams<U>, Condition = URhs>,
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
        let mut theta = HashMap::new();
        theta.insert(1u32, NormalParams::new(1.0, 2.0).unwrap());
        theta.insert(2u32, NormalParams::new(2.0, 2.0).unwrap());
        theta.insert(3u32, NormalParams::new(3.0, 2.0).unwrap());
        theta.insert(4u32, NormalParams::new(4.0, 2.0).unwrap());
        let distr = Normal.switch(&theta);
        let switched_fk = distr.fk(&0f64, &SwitchedParams::Key(1u32)).unwrap();
        let fk = Normal
            .fk(&0f64, &NormalParams::new(1.0, 2.0).unwrap())
            .unwrap();

        assert_eq!(switched_fk, fk);
    }
}
