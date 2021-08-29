use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::StdRng;
use std::{
    fmt::Debug,
    ops::{BitAnd, Mul},
};

#[derive(Clone)]
pub struct ConvertedDistribution<'a, D, T1, T2, U>
where
    D: Distribution<T = T1, U = U>,
    T1: RandomVariable,
    T2: RandomVariable,
    U: RandomVariable,
{
    distribution: D,
    map: &'a (dyn Fn(T1) -> Result<T2, DistributionError> + Send + Sync),
    inv: &'a (dyn Fn(&T2) -> Result<T1, DistributionError> + Send + Sync),
}

impl<'a, D, T1, T2, U> ConvertedDistribution<'a, D, T1, T2, U>
where
    D: Distribution<T = T1, U = U>,
    T1: RandomVariable,
    T2: RandomVariable,
    U: RandomVariable,
{
    pub fn new(
        distribution: D,
        map: &'a (dyn Fn(T1) -> Result<T2, DistributionError> + Send + Sync),
        inv: &'a (dyn Fn(&T2) -> Result<T1, DistributionError> + Send + Sync),
    ) -> Self {
        Self {
            distribution,
            map,
            inv,
        }
    }
}

impl<'a, D, T1, T2, U> Debug for ConvertedDistribution<'a, D, T1, T2, U>
where
    D: Distribution<T = T1, U = U>,
    T1: RandomVariable,
    T2: RandomVariable,
    U: RandomVariable,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ConvertedDistribution {{ distribution: {:#?} }}",
            self.distribution
        )
    }
}

impl<'a, D, T1, T2, U> Distribution for ConvertedDistribution<'a, D, T1, T2, U>
where
    D: Distribution<T = T1, U = U>,
    T1: RandomVariable,
    T2: RandomVariable,
    U: RandomVariable,
{
    type T = T2;
    type U = U;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        self.distribution.p(&(self.inv)(x)?, theta)
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, DistributionError> {
        let value = self.distribution.sample(theta, rng)?;

        Ok((self.map)(value)?)
    }
}

pub trait ConvertableDistribution: Distribution + Sized {
    fn convert<'a, T2>(
        self,
        map: &'a (dyn Fn(Self::T) -> Result<T2, DistributionError> + Send + Sync),
        inv: &'a (dyn Fn(&T2) -> Result<Self::T, DistributionError> + Send + Sync),
    ) -> ConvertedDistribution<'a, Self, Self::T, T2, Self::U>
    where
        T2: RandomVariable;
}

impl<D, T1, U> ConvertableDistribution for D
where
    D: Distribution<T = T1, U = U>,
    T1: RandomVariable,
    U: RandomVariable,
{
    fn convert<'a, T2>(
        self,
        map: &'a (dyn Fn(Self::T) -> Result<T2, DistributionError> + Send + Sync),
        inv: &'a (dyn Fn(&T2) -> Result<Self::T, DistributionError> + Send + Sync),
    ) -> ConvertedDistribution<'a, Self, Self::T, T2, Self::U>
    where
        T2: RandomVariable,
    {
        ConvertedDistribution::<Self, Self::T, T2, Self::U>::new(self, map, inv)
    }
}

impl<'a, D, T1, T2, U, Rhs, TRhs> Mul<Rhs> for ConvertedDistribution<'a, D, T1, T2, U>
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

impl<'a, D, T1, T2, U, Rhs, URhs> BitAnd<Rhs> for ConvertedDistribution<'a, D, T1, T2, U>
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

#[cfg(test)]
mod tests {
    use crate::distribution::Distribution;
    use crate::*;
    use rand::prelude::*;
    #[test]
    fn it_works() {
        let model = Cauchy.convert(&|x| CauchyParams::new(1.0, x), &|theta| Ok(theta.sigma()));
        let mut rng = StdRng::from_seed([1; 32]);

        let x = model
            .sample(&CauchyParams::new(0.0, 1.0).unwrap(), &mut rng)
            .unwrap();

        println!("{:#?}", x);
    }
}
