use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::StdRng;
use std::iter::Iterator;
use std::{ops::BitAnd, ops::Mul};

/// p(x1, …, xn) = Π p(xi)
#[derive(Clone, Debug)]
pub struct IndependentArrayJoint<D, T, U>
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    distributions: Vec<D>,
}

impl<D, T, U> Distribution for IndependentArrayJoint<D, T, U>
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    type T = Vec<T>;
    type U = U;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        x.iter()
            .enumerate()
            .map(|(i, xi)| self.distributions[i].p(xi, theta))
            .product()
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, DistributionError> {
        self.distributions
            .iter()
            .map(|di| di.sample(theta, rng))
            .collect()
    }
}

impl<D, T, U, Rhs, TRhs> Mul<Rhs> for IndependentArrayJoint<D, T, U>
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
    Rhs: Distribution<T = TRhs, U = U>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, Vec<T>, TRhs, U>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<D, T, U, Rhs, URhs> BitAnd<Rhs> for IndependentArrayJoint<D, T, U>
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
    Rhs: Distribution<T = U, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, Vec<T>, U, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

pub trait DistributionProduct<D, T, U>
where
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    fn joint(self) -> IndependentArrayJoint<D, T, U>;
}

impl<I, D, T, U> DistributionProduct<D, T, U> for I
where
    I: Iterator<Item = D>,
    D: Distribution<T = T, U = U>,
    T: RandomVariable,
    U: RandomVariable,
{
    fn joint(self) -> IndependentArrayJoint<D, T, U> {
        let distributions = self.collect::<Vec<_>>();

        IndependentArrayJoint::<D, T, U> { distributions }
    }
}

#[cfg(test)]
mod tests {
    use crate::distribution::Distribution;
    use crate::*;
    use rand::prelude::*;
    #[test]
    fn it_works() {
        let model = vec![Normal; 3].into_iter().joint();

        let mut rng = StdRng::from_seed([1; 32]);

        let x = model
            .sample(&NormalParams::new(0.0, 1.0).unwrap(), &mut rng)
            .unwrap();

        println!("{:#?}", x);
    }
}
