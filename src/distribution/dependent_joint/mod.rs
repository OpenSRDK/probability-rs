use crate::{Distribution, IndependentJoint, RandomVariable};
use crate::{DistributionError, Event};
use rand::prelude::*;
use std::{ops::BitAnd, ops::Mul};

/// p(x, y) = p(x | y) p(y)
#[derive(Clone, Debug)]
pub struct DependentJoint<L, R, T, UL, UR>
where
    L: Distribution<Value = T, Condition = UL>,
    R: Distribution<Value = UL, Condition = UR>,
    T: RandomVariable,
    UL: Event,
    UR: Event,
{
    lhs: L,
    rhs: R,
}

impl<L, R, T, UL, UR> DependentJoint<L, R, T, UL, UR>
where
    L: Distribution<Value = T, Condition = UL>,
    R: Distribution<Value = UL, Condition = UR>,
    T: RandomVariable,
    UL: Event,
    UR: Event,
{
    pub fn new(lhs: L, rhs: R) -> Self {
        Self { lhs, rhs }
    }
}

impl<L, R, T, UL, UR> Distribution for DependentJoint<L, R, T, UL, UR>
where
    L: Distribution<Value = T, Condition = UL>,
    R: Distribution<Value = UL, Condition = UR>,
    T: RandomVariable,
    UL: RandomVariable,
    UR: Event,
{
    type Value = (T, UL);
    type Condition = UR;

    fn fk(&self, x: &(T, UL), theta: &UR) -> Result<f64, DistributionError> {
        Ok(self.lhs.fk(&x.0, &x.1)? * self.rhs.fk(&x.1, theta)?)
    }

    fn sample(&self, theta: &UR, rng: &mut dyn RngCore) -> Result<(T, UL), DistributionError> {
        let rhs = self.rhs.sample(theta, rng)?;
        Ok((self.lhs.sample(&rhs, rng)?, rhs))
    }
}

impl<L, R, T, UL, UR, Rhs, TRhs> Mul<Rhs> for DependentJoint<L, R, T, UL, UR>
where
    L: Distribution<Value = T, Condition = UL>,
    R: Distribution<Value = UL, Condition = UR>,
    T: RandomVariable,
    UL: RandomVariable,
    UR: RandomVariable,
    Rhs: Distribution<Value = TRhs, Condition = UR>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, (T, UL), TRhs, UR>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<L, R, T, UL, UR, Rhs, URhs> BitAnd<Rhs> for DependentJoint<L, R, T, UL, UR>
where
    L: Distribution<Value = T, Condition = UL>,
    R: Distribution<Value = UL, Condition = UR>,
    T: RandomVariable,
    UL: RandomVariable,
    UR: RandomVariable,
    Rhs: Distribution<Value = UR, Condition = URhs>,
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
        let model = Normal.condition(|x: &f64| NormalParams::new(1.0, x.powi(2) + 1.0)) & Normal;
        let mut rng = StdRng::from_seed([1; 32]);

        let x = model
            .sample(&NormalParams::new(0.0, 1.0).unwrap(), &mut rng)
            .unwrap();

        println!("{:#?}", x);
    }
}
