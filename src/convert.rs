use std::{
    error::Error,
    ops::{BitAnd, Mul},
};

use rand::prelude::StdRng;

use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};

pub struct ConvertedDistribution<D, T1, T2, U>
where
    D: Distribution<T = T1, U = U>,
    T1: RandomVariable,
    T2: RandomVariable,
    U: RandomVariable,
{
    distribution: D,
    map: Box<dyn Fn(T1) -> T2>,
    inv: Box<dyn Fn(&T2) -> T1>,
}

impl<D, T1, T2, U> ConvertedDistribution<D, T1, T2, U>
where
    D: Distribution<T = T1, U = U>,
    T1: RandomVariable,
    T2: RandomVariable,
    U: RandomVariable,
{
    pub fn new(distribution: D, map: Box<dyn Fn(T1) -> T2>, inv: Box<dyn Fn(&T2) -> T1>) -> Self {
        Self {
            distribution,
            map,
            inv,
        }
    }
}

impl<D, T1, T2, U> Distribution for ConvertedDistribution<D, T1, T2, U>
where
    D: Distribution<T = T1, U = U>,
    T1: RandomVariable,
    T2: RandomVariable,
    U: RandomVariable,
{
    type T = T2;
    type U = U;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, Box<dyn Error>> {
        self.distribution.p(&(self.inv)(x), theta)
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, Box<dyn Error>> {
        let value = self.distribution.sample(theta, rng)?;

        Ok((self.map)(value))
    }
}

pub trait ConvertableDistribution: Distribution + Sized {
    fn convert<T2>(
        self,
        map: Box<dyn Fn(Self::T) -> T2>,
        inv: Box<dyn Fn(&T2) -> Self::T>,
    ) -> ConvertedDistribution<Self, Self::T, T2, Self::U>
    where
        T2: RandomVariable;
}

impl<D, T1, U> ConvertableDistribution for D
where
    D: Distribution<T = T1, U = U>,
    T1: RandomVariable,
    U: RandomVariable,
{
    fn convert<T2>(
        self,
        map: Box<dyn Fn(Self::T) -> T2>,
        inv: Box<dyn Fn(&T2) -> Self::T>,
    ) -> ConvertedDistribution<Self, Self::T, T2, Self::U>
    where
        T2: RandomVariable,
    {
        ConvertedDistribution::<Self, Self::T, T2, Self::U>::new(self, map, inv)
    }
}

impl<D, T1, T2, U, Rhs, TRhs> Mul<Rhs> for ConvertedDistribution<D, T1, T2, U>
where
    D: Distribution<T = T1, U = U>,
    T1: RandomVariable,
    T2: RandomVariable,
    U: RandomVariable,
    Rhs: Distribution<T = TRhs, U = U>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, T2, TRhs, U>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<D, T1, T2, U, Rhs, URhs> BitAnd<Rhs> for ConvertedDistribution<D, T1, T2, U>
where
    D: Distribution<T = T1, U = U>,
    T1: RandomVariable,
    T2: RandomVariable,
    U: RandomVariable,
    Rhs: Distribution<T = U, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, T2, U, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}
