use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use std::{
    collections::HashMap,
    fmt::Debug,
    ops::{BitAnd, Mul},
};

#[derive(Clone, Debug)]
pub struct SwitchDistribution<'a, D, T, U>
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    distribution: D,
    map: &'a HashMap<u32, U>,
    default: U,
}

impl<'a, D, T, U> SwitchDistribution<'a, D, T, U>
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    pub fn new(distribution: D, map: &'a HashMap<u32, U>, default: U) -> Self {
        Self {
            distribution,
            map,
            default,
        }
    }
}

impl<'a, D, T, U> Distribution for SwitchDistribution<'a, D, T, U>
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    type T = T;
    type U = u32;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let s = theta;

        match self.map.get(s) {
            Some(theta) => self.distribution.p(x, &theta),
            None => self.distribution.p(x, &self.default),
        }
    }

    fn sample(&self, theta: &Self::U, rng: &mut dyn RngCore) -> Result<Self::T, DistributionError> {
        let s = theta;

        match self.map.get(s) {
            Some(theta) => self.distribution.sample(&theta, rng),
            None => self.distribution.sample(&self.default, rng),
        }
    }
}

pub trait SwitchableDistribution<U>: Distribution + Sized
where
    U: RandomVariable,
{
    fn switch<'a>(
        self,
        map: &'a HashMap<u32, U>,
        default: U,
    ) -> SwitchDistribution<'a, Self, Self::T, Self::U>;
}

impl<D, T, U> SwitchableDistribution<U> for D
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    fn switch<'a>(
        self,
        map: &'a HashMap<u32, U>,
        default: U,
    ) -> SwitchDistribution<'a, Self, Self::T, Self::U> {
        SwitchDistribution::<Self, Self::T, Self::U>::new(self, map, default)
    }
}

impl<'a, D, T, U, Rhs, TRhs> Mul<Rhs> for SwitchDistribution<'a, D, T, U>
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
    Rhs: Distribution<T = TRhs, U = u32>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, T, TRhs, u32>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<'a, D, T, U, Rhs, URhs> BitAnd<Rhs> for SwitchDistribution<'a, D, T, U>
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
    Rhs: Distribution<T = u32, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, T, u32, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}
