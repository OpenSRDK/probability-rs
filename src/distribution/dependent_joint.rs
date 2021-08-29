use crate::DistributionError;
use rand::prelude::StdRng;

use crate::{Distribution, IndependentJoint, RandomVariable};
use std::{ops::BitAnd, ops::Mul};

/// # DependentJoint
/// ![tex](https://latex.codecogs.com/svg.latex?p%28a,b%7Cc%29%3Dp%28a%7Cb%29p%28b%7Cc%29)
#[derive(Clone, Debug)]
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
        Self { lhs, rhs }
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

    fn p(&self, x: &(T, UL), theta: &UR) -> Result<f64, DistributionError> {
        Ok(self.lhs.p(&x.0, &x.1)? * self.rhs.p(&x.1, theta)?)
    }

    fn sample(&self, theta: &UR, rng: &mut StdRng) -> Result<(T, UL), DistributionError> {
        let rhs = self.rhs.sample(theta, rng)?;
        Ok((self.lhs.sample(&rhs, rng)?, rhs))
    }
}

impl<L, R, T, UL, UR, Rhs, TRhs> Mul<Rhs> for DependentJoint<L, R, T, UL, UR>
where
    L: Distribution<T = T, U = UL>,
    R: Distribution<T = UL, U = UR>,
    T: RandomVariable,
    UL: RandomVariable,
    UR: RandomVariable,
    Rhs: Distribution<T = TRhs, U = UR>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, (T, UL), TRhs, UR>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<L, R, T, UL, UR, Rhs, URhs> BitAnd<Rhs> for DependentJoint<L, R, T, UL, UR>
where
    L: Distribution<T = T, U = UL>,
    R: Distribution<T = UL, U = UR>,
    T: RandomVariable,
    UL: RandomVariable,
    UR: RandomVariable,
    Rhs: Distribution<T = UR, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, (T, UL), UR, URhs>;

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
        let model =
            Cauchy & Cauchy.convert(&|x| CauchyParams::new(1.0, x), &|theta| Ok(theta.sigma()));
        let mut rng = StdRng::from_seed([1; 32]);

        let x = model
            .sample(&CauchyParams::new(0.0, 1.0).unwrap(), &mut rng)
            .unwrap();

        println!("{:#?}", x);
    }
}
