use crate::{
    ConditionDifferentiableDistribution, Distribution, IndependentJoint, RandomVariable,
    ValueDifferentiableDistribution,
};
use crate::{DistributionError, Event};
use opensrdk_linear_algebra::Vector;
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
    pub(crate) lhs: L,
    pub(crate) rhs: R,
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

impl<L, R, T, UL, UR> ValueDifferentiableDistribution for DependentJoint<L, R, T, UL, UR>
where
    L: Distribution<Value = T, Condition = UL> + ValueDifferentiableDistribution,
    R: Distribution<Value = UL, Condition = UR> + ValueDifferentiableDistribution,
    T: RandomVariable,
    UL: RandomVariable,
    UR: RandomVariable,
{
    fn ln_diff_value(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, crate::DistributionError> {
        let diff_l = &self.lhs.ln_diff_value(&x.0, &x.1)?;
        let diff = (diff_l.clone().col_mat() * &self.rhs.fk(&x.1, theta)?).vec();
        Ok(diff)
    }
}

impl<L, R, T, UL, UR> ConditionDifferentiableDistribution for DependentJoint<L, R, T, UL, UR>
where
    L: Distribution<Value = T, Condition = UL> + ConditionDifferentiableDistribution,
    R: Distribution<Value = UL, Condition = UR> + ConditionDifferentiableDistribution,
    T: RandomVariable,
    UL: RandomVariable,
    UR: RandomVariable,
{
    fn ln_diff_condition(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, crate::DistributionError> {
        let diff_r = &self.rhs.ln_diff_condition(&x.1, &theta)?;
        let diff = (&self.lhs.fk(&x.0, &x.1)? * diff_r.clone().col_mat()).vec();
        Ok(diff)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use crate::{
        ConditionDifferentiableDistribution, Distribution, ExactMultivariateNormalParams,
        MultivariateNormal, ValueDifferentiableDistribution,
    };
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

    #[test]
    fn it_works2() {
        let model = Normal.condition(|x: &f64| NormalParams::new(1.0, x.powi(2) + 1.0)) & Normal;

        let f = model
            .ln_diff_value(&(1.0, 2.0), &NormalParams::new(0.0, 1.0).unwrap())
            .unwrap();

        println!("{:#?}", f);
    }

    #[test]
    fn it_works3() {
        let model_prior = Normal.condition(|x: &f64| NormalParams::new(1.0, x.powi(2) + 1.0));
        let g = |theta: &f64| Ok(vec![0.0, 2.0 * theta]);
        let model = ConditionDifferentiableConditionedDistribution::new(model_prior, g) & Normal;

        let f = model
            .ln_diff_condition(&(1.0, 2.0), &NormalParams::new(0.0, 1.0).unwrap())
            .unwrap();

        println!("{:#?}", f);
    }
}
