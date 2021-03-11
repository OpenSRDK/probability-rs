use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::StdRng;
use std::{
    error::Error,
    fmt::Debug,
    ops::{BitAnd, Mul},
};

#[derive(Clone)]
pub struct InstantDistribution<'a, T, U>
where
    T: RandomVariable,
    U: RandomVariable,
{
    p: &'a dyn Fn(&T, &U) -> Result<f64, Box<dyn Error>>,
    sample: &'a dyn Fn(&U, &mut StdRng) -> Result<T, Box<dyn Error>>,
}

impl<'a, T, U> InstantDistribution<'a, T, U>
where
    T: RandomVariable,
    U: RandomVariable,
{
    pub fn new(
        p: &'a dyn Fn(&T, &U) -> Result<f64, Box<dyn Error>>,
        sample: &'a dyn Fn(&U, &mut StdRng) -> Result<T, Box<dyn Error>>,
    ) -> Self {
        Self { p, sample }
    }
}

impl<'a, T, U> Debug for InstantDistribution<'a, T, U>
where
    T: RandomVariable,
    U: RandomVariable,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Instant")
    }
}

impl<'a, T, U> Distribution for InstantDistribution<'a, T, U>
where
    T: RandomVariable,
    U: RandomVariable,
{
    type T = T;
    type U = U;

    fn p(&self, x: &T, theta: &U) -> Result<f64, Box<dyn Error>> {
        (self.p)(x, theta)
    }

    fn sample(&self, theta: &U, rng: &mut StdRng) -> Result<T, Box<dyn Error>> {
        (self.sample)(theta, rng)
    }
}

impl<'a, T, U, Rhs, TRhs> Mul<Rhs> for InstantDistribution<'a, T, U>
where
    T: RandomVariable,
    U: RandomVariable,
    Rhs: Distribution<T = TRhs, U = U>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, T, TRhs, U>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<'a, T, U, Rhs, URhs> BitAnd<Rhs> for InstantDistribution<'a, T, U>
where
    T: RandomVariable,
    U: RandomVariable,
    Rhs: Distribution<T = U, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, T, U, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}
