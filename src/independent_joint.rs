use rand::prelude::StdRng;

use crate::{Distribution, RandomVariable};
use std::{error::Error, marker::PhantomData};

/// # IndependentJoint
/// ![tex](https://latex.codecogs.com/svg.latex?P%28a,b%7Cc%29%3DP%28a%7Cc%29P%28b%7Cc%29)
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
