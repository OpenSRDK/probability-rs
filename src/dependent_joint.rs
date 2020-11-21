use rand::prelude::StdRng;

use crate::{Distribution, RandomVariable};
use std::{error::Error, marker::PhantomData};

/// # DependentJoint
/// ![tex](https://latex.codecogs.com/svg.latex?P%28a,b%7Cc%29%3DP%28a%7Cb%29P%28b%7Cc%29)
pub struct DependentJoint<L, R, T, UL, UR>
where
    L: Distribution<T = T, U = UL>,
    R: Distribution<T = UL, U = UR>,
    T: RandomVariable,
    UL: RandomVariable,
    UR: RandomVariable,
{
    lhs: L,
    rhs: R,
    phantom: PhantomData<(T, UL, UR)>,
}

impl<L, R, T, UL, UR> DependentJoint<L, R, T, UL, UR>
where
    L: Distribution<T = T, U = UL>,
    R: Distribution<T = UL, U = UR>,
    T: RandomVariable,
    UL: RandomVariable,
    UR: RandomVariable,
{
    pub fn new(lhs: L, rhs: R) -> Self {
        Self {
            lhs,
            rhs,
            phantom: PhantomData,
        }
    }
}

impl<L, R, T, UL, UR> Distribution for DependentJoint<L, R, T, UL, UR>
where
    L: Distribution<T = T, U = UL>,
    R: Distribution<T = UL, U = UR>,
    T: RandomVariable,
    UL: RandomVariable,
    UR: RandomVariable,
{
    type T = (T, UL);
    type U = UR;

    fn p(&self, x: &(T, UL), theta: &UR) -> Result<f64, Box<dyn Error>> {
        Ok(self.lhs.p(&x.0, &x.1)? * self.rhs.p(&x.1, theta)?)
    }

    fn sample(&self, theta: &UR, rng: &mut StdRng) -> Result<(T, UL), Box<dyn Error>> {
        let rhs = self.rhs.sample(theta, rng)?;
        Ok((self.lhs.sample(&rhs, rng)?, rhs))
    }
}
