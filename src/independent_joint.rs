use crate::{DependentJoint, Distribution, RandomVariable};
use rand::prelude::StdRng;
use std::{error::Error, marker::PhantomData, ops::BitAnd, ops::Mul};

/// # IndependentJoint
/// ![tex](https://latex.codecogs.com/svg.latex?p%28a,b%7Cc%29%3Dp%28a%7Cc%29p%28b%7Cc%29)
pub struct IndependentJoint<L, R, TL, TR, U>
where
    L: Distribution<T = TL, U = U>,
    R: Distribution<T = TR, U = U>,
    TL: RandomVariable,
    TR: RandomVariable,
    U: RandomVariable,
{
    lhs: L,
    rhs: R,
    phantom: PhantomData<(TL, TR, U)>,
}

impl<L, R, TL, TR, U> IndependentJoint<L, R, TL, TR, U>
where
    L: Distribution<T = TL, U = U>,
    R: Distribution<T = TR, U = U>,
    TL: RandomVariable,
    TR: RandomVariable,
    U: RandomVariable,
{
    pub fn new(lhs: L, rhs: R) -> Self {
        Self {
            lhs,
            rhs,
            phantom: PhantomData,
        }
    }
}

impl<L, R, TL, TR, U> Distribution for IndependentJoint<L, R, TL, TR, U>
where
    L: Distribution<T = TL, U = U>,
    R: Distribution<T = TR, U = U>,
    TL: RandomVariable,
    TR: RandomVariable,
    U: RandomVariable,
{
    type T = (TL, TR);
    type U = U;

    fn p(&self, x: &(TL, TR), theta: &U) -> Result<f64, Box<dyn Error>> {
        Ok(self.lhs.p(&x.0, theta)? * self.rhs.p(&x.1, theta)?)
    }

    fn sample(&self, theta: &U, rng: &mut StdRng) -> Result<(TL, TR), Box<dyn Error>> {
        Ok((self.lhs.sample(theta, rng)?, self.rhs.sample(theta, rng)?))
    }
}

impl<L, R, TL, TR, U, Rhs, TRhs> Mul<Rhs> for IndependentJoint<L, R, TL, TR, U>
where
    L: Distribution<T = TL, U = U>,
    R: Distribution<T = TR, U = U>,
    TL: RandomVariable,
    TR: RandomVariable,
    U: RandomVariable,
    Rhs: Distribution<T = TRhs, U = U>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, (TL, TR), TRhs, U>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<L, R, TL, TR, U, Rhs, URhs> BitAnd<Rhs> for IndependentJoint<L, R, TL, TR, U>
where
    L: Distribution<T = TL, U = U>,
    R: Distribution<T = TR, U = U>,
    TL: RandomVariable,
    TR: RandomVariable,
    U: RandomVariable,
    Rhs: Distribution<T = U, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, (TL, TR), U, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}
