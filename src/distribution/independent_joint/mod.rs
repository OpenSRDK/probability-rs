use crate::{
    ConditionDifferentiableDistribution, DependentJoint, Distribution, RandomVariable,
    ValueDifferentiableDistribution,
};
use crate::{DistributionError, Event};
use rand::prelude::*;
use std::fmt::Debug;
use std::{ops::BitAnd, ops::Mul};

/// p(x, y) = p(x) p(y)
#[derive(Clone, Debug)]
pub struct IndependentJoint<L, R, TL, TR, U>
where
    L: Distribution<Value = TL, Condition = U>,
    R: Distribution<Value = TR, Condition = U>,
    TL: RandomVariable,
    TR: RandomVariable,
    U: Event,
{
    lhs: L,
    rhs: R,
}

impl<L, R, TL, TR, U> IndependentJoint<L, R, TL, TR, U>
where
    L: Distribution<Value = TL, Condition = U>,
    R: Distribution<Value = TR, Condition = U>,
    TL: RandomVariable,
    TR: RandomVariable,
    U: Event,
{
    pub fn new(lhs: L, rhs: R) -> Self {
        Self { lhs, rhs }
    }
}

impl<L, R, TL, TR, U> Distribution for IndependentJoint<L, R, TL, TR, U>
where
    L: Distribution<Value = TL, Condition = U>,
    R: Distribution<Value = TR, Condition = U>,
    TL: RandomVariable,
    TR: RandomVariable,
    U: Event,
{
    type Value = (TL, TR);
    type Condition = U;

    fn fk(&self, x: &(TL, TR), theta: &U) -> Result<f64, DistributionError> {
        Ok(self.lhs.fk(&x.0, theta)? * self.rhs.fk(&x.1, theta)?)
    }

    fn sample(&self, theta: &U, rng: &mut dyn RngCore) -> Result<(TL, TR), DistributionError> {
        Ok((self.lhs.sample(theta, rng)?, self.rhs.sample(theta, rng)?))
    }
}

impl<L, R, TL, TR, U, Rhs, TRhs> Mul<Rhs> for IndependentJoint<L, R, TL, TR, U>
where
    L: Distribution<Value = TL, Condition = U>,
    R: Distribution<Value = TR, Condition = U>,
    TL: RandomVariable,
    TR: RandomVariable,
    U: Event,
    Rhs: Distribution<Value = TRhs, Condition = U>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, (TL, TR), TRhs, U>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<L, R, TL, TR, U, Rhs, URhs> BitAnd<Rhs> for IndependentJoint<L, R, TL, TR, U>
where
    L: Distribution<Value = TL, Condition = U>,
    R: Distribution<Value = TR, Condition = U>,
    TL: RandomVariable,
    TR: RandomVariable,
    U: Event,
    Rhs: Distribution<Value = U, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, (TL, TR), U, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

impl<L, R, TL, TR, U> ValueDifferentiableDistribution for IndependentJoint<L, R, TL, TR, U>
where
    L: Distribution<Value = TL, Condition = U> + ValueDifferentiableDistribution,
    R: Distribution<Value = TR, Condition = U> + ValueDifferentiableDistribution,
    TL: RandomVariable,
    TR: RandomVariable,
    U: Event,
{
    fn ln_diff_value(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let f_lhs = self.lhs.ln_diff_value(&x.0, theta)?;
        let f_rhs = self.rhs.ln_diff_value(&x.1, theta)?;
        Ok([f_lhs, f_rhs].concat())
    }
}

impl<L, R, TL, TR, U> ConditionDifferentiableDistribution for IndependentJoint<L, R, TL, TR, U>
where
    L: Distribution<Value = TL, Condition = U> + ConditionDifferentiableDistribution,
    R: Distribution<Value = TR, Condition = U> + ConditionDifferentiableDistribution,
    TL: RandomVariable,
    TR: RandomVariable,
    U: Event,
{
    fn ln_diff_condition(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let f_lhs = self.lhs.ln_diff_condition(&x.0, theta)?;
        let f_rhs = self.rhs.ln_diff_condition(&x.1, theta)?;
        let f = f_lhs
            .iter()
            .enumerate()
            .map(|(i, f_lhsi)| f_lhsi + f_rhs[i])
            .collect::<Vec<f64>>();
        Ok(f)
    }
}
#[cfg(test)]
mod tests {
    use crate::distribution::Distribution;
    use crate::*;
    use rand::prelude::*;
    #[test]
    fn it_works() {
        let model = Normal * Normal;

        let mut rng = StdRng::from_seed([1; 32]);

        let x = model
            .sample(&NormalParams::new(0.0, 1.0).unwrap(), &mut rng)
            .unwrap();

        println!("{:#?}", x);
    }
}
